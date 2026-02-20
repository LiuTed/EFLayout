from typing import Tuple, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.distributions import Categorical

from configuration import EFLayoutConfig, EFLayoutGraphConfig, EFLayoutSpatialConfig



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class EFLayoutSpatialAttention(nn.Module):
    def __init__(self, config: EFLayoutSpatialConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seqlen, _ = hidden_states.shape
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seqlen, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(seqlen, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output
           

class EFLayoutSpatialMLP(nn.Module):
    def __init__(self, config: EFLayoutSpatialConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))     


class EFLayoutSpatialBlock(nn.Module):
    def __init__(self, config: EFLayoutSpatialConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = EFLayoutSpatialAttention(config)
        self.mlp = EFLayoutSpatialMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attn_mask=attn_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class EFLayoutRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class EFLayoutSpatialModel(nn.Module):
    def __init__(self, config: EFLayoutSpatialConfig, id_emb: nn.Embedding):
        super().__init__()
        self.id_emb = id_emb
        self.type_emb = nn.Embedding(config.type_num, config.type_dim)
        self.state_emb = nn.Linear(config.state_dim, config.hidden_size - config.type_dim - config.id_size)
        dim = config.hidden_size // config.num_heads
        self.rot_pos_emb = EFLayoutRotaryEmbedding(dim // 2, config.theta)

        self.blocks = nn.ModuleList([
            EFLayoutSpatialBlock(config)
            for _ in range(config.num_layers)
        ])
        self.out_proj = nn.Linear(config.hidden_size, config.output_dim)

        self.config = config
    
    def get_position_embeddings(self, spatial_hw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial_hw.ndim == 1:
            spatial_hw = spatial_hw.unsqueeze(0)

        max_hw = int(spatial_hw.max().item())
        freq_table = self.rot_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(spatial_hw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for height, width in spatial_hw:
            block_rows = torch.arange(height, device=device)  # block row indices
            block_cols = torch.arange(width, device=device)  # block col indices

            # Compute full-resolution positions
            row_idx = block_rows.unsqueeze(-1).expand(height, width).reshape(-1)
            col_idx = block_cols.unsqueeze(0).expand(height, width).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings
    
    def get_attn_mask(self, spatial_hw: torch.Tensor) -> torch.Tensor:
        cu_seqlens = spatial_hw.prod(-1)
        blocks = [torch.ones((n, n), dtype=torch.bool, device=spatial_hw.device) for n in cu_seqlens]
        return torch.block_diag(*blocks)
    
    def forward(self, spatial_info: Dict[str, torch.Tensor], spatial_hw: torch.Tensor) -> torch.Tensor:
        id_emb = self.id_emb(spatial_info['ids'] % self.id_emb.num_embeddings)
        
        type_emb = self.type_emb(spatial_info['types'])

        rotary_pos_emb = self.get_position_embeddings(spatial_hw)
        hidden_states = self.state_emb(spatial_info['states'])

        hidden_states = torch.cat([id_emb, type_emb, hidden_states], dim=-1)
        
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        attn_mask = self.get_attn_mask(spatial_hw)
        
        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask, position_embeddings)
        
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
        
    
class EFLayoutGraphBlock(nn.Module):
    def __init__(self, config: EFLayoutGraphConfig, layer_idx: int):
        super().__init__()
        self.edge_dim = None
        if config.edge_dim > 0:
            self.edge_dim = config.edge_dim
        if layer_idx == 0:
            self.attn = GATv2Conv(
                config.state_dim + config.type_dim + config.id_size,
                config.hidden_size,
                heads=config.num_heads,
                edge_dim=self.edge_dim,
                dropout=config.dropout,
                residual=True,
                concat=True
            )
        else:
            dim = config.hidden_size * config.num_heads
            self.attn = GATv2Conv(
                dim,
                dim,
                heads=config.num_heads,
                edge_dim=self.edge_dim,
                dropout=config.dropout,
                residual=True,
                concat=False
            )
        self.config = config
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if edge_attrs is not None:
            assert self.edge_dim is not None, f"edge_dim is None, unexpected edge_attrs={edge_attrs}"
        return F.elu(self.attn(hidden_states, edge_index, edge_attr=edge_attrs))


class EFLayoutGraphModel(nn.Module):
    def __init__(self, config: EFLayoutGraphConfig, id_emb: nn.Embedding):
        super().__init__()
        self.id_emb = id_emb
        self.type_emb = nn.Embedding(config.type_num, config.type_dim)
        self.blocks = nn.ModuleList([
            EFLayoutGraphBlock(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        self.out_proj = nn.Linear(config.num_heads * config.hidden_size, config.output_dim)

    def forward(
        self,
        graph_nodes: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attrs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        id_emb = self.id_emb(graph_nodes['ids'] % self.id_emb.num_embeddings)
        type_emb = self.type_emb(graph_nodes['types'])
        hidden_states = torch.cat([id_emb, type_emb, graph_nodes['states']], dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states, edge_index, edge_attrs)

        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class EFLayoutACContext(nn.Module):
    def __init__(self, config: EFLayoutConfig):
        super().__init__()

    def forward(self, spatial_embeds: torch.Tensor, graph_embeds: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_proj(spatial_embeds)
        graph = self.graph_proj(graph_embeds)
        context = torch.cat([spatial, graph], dim=-1)
        return F.relu(self.proj(context)), spatial, graph


class EFLayoutModelBackbone(nn.Module):
    def __init__(self, config: EFLayoutConfig):
        super().__init__()
        self.config = config

        self.id_emb = nn.Embedding(config.id_buckets, config.id_size)
        self.spatial = EFLayoutSpatialModel(config.spatial_config, self.id_emb)
        self.graph = EFLayoutGraphModel(config.graph_config, self.id_emb)
        self.spatial_proj = nn.Linear(config.spatial_config.output_dim, config.hidden_size)
        self.graph_proj = nn.Linear(config.graph_config.output_dim, config.hidden_size)
        self.context_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        spatial_info: Dict[str, torch.Tensor],
        spatial_hw: torch.Tensor, # (B, 2)
        graph_nodes: Dict[str, torch.Tensor], # (sum N_n, D_g)
        edge_index: torch.Tensor, # (2, sum N_e)
        node_lengths: torch.Tensor, # (B)
        edge_attrs: Optional[torch.Tensor] = None, # (sum N_e, D_e)
        **kwargs,
    ):
        spatial_embeds = self.spatial(spatial_info, spatial_hw)
        graph_embeds = self.graph(graph_nodes, edge_index, edge_attrs)
        
        spatial = self.spatial_proj(spatial_embeds)
        graph = self.graph_proj(graph_embeds)
        
        spatial_info_length = spatial_hw.prod(-1)
        spatial_avg = []
        for part in spatial.split(spatial_info_length.tolist()):
            spatial_avg.append(part.mean(dim=0))
        spatial_avg = torch.stack(spatial_avg, dim=0)

        graph_avg = []
        for part in graph.split(node_lengths.tolist()):
            graph_avg.append(part.mean(dim=0))
        graph_avg = torch.stack(graph_avg, dim=0)

        context = torch.cat([spatial_avg, graph_avg], dim=-1)
        context = self.context_proj(context)

        return {
            'context': context, # (B, D)
            'spatial_embeds': spatial, # (sum M*N, D)
            'graph_embeds': graph, # (sum N_n, D)
            'spatial_embeds_avg': spatial_avg, # (B, D)
            'graph_embeds_avg': graph_avg, # (B, D)
            'spatial_hw': spatial_hw,
            'node_lengths': node_lengths,
        }
        

class EFLayoutActorCritic(nn.Module):
    def __init__(self, config: EFLayoutConfig):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )
        # action select
        self.action_selector = nn.Linear(config.hidden_size, 3)
        # machine
        ## machine select
        self.ms_q_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.ms_k_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        ## machine location & orientation
        self.mlo_q_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlo_k_proj = nn.Linear(config.hidden_size, config.intermediate_size * 4, bias=False)
        # belt
        self.b_q_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.b_k_proj = nn.Linear(config.hidden_size, config.intermediate_size * 4, bias=False)
        # power supply
        self.p_q_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.p_k_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        
        self.config = config

    def get_critic(self, context: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(context)
    
    def get_action(self, context: torch.Tensor, action_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.action_selector(context)
        logits[action_mask.reshape(-1, 3).logical_not()] = -float('inf')
        return logits.flatten(), logits.unbind(0)
    
    def get_machine(
        self,
        context: torch.Tensor, # (B, D)
        graph_embeds: torch.Tensor, # (sum N_n, D)
        machine_mask: torch.Tensor, # (sum N_n)
        node_lengths: torch.Tensor, # (B)
        **kwargs
    ) -> torch.Tensor:
        msq = self.ms_q_proj(context)
        msk = self.ms_k_proj(graph_embeds)

        all_logits = []
        for q, k, mask in zip(
            msq,
            msk.split(node_lengths.tolist()),
            machine_mask.split(node_lengths.tolist())
        ):
            logits = torch.inner(q, k) # (D) . (N_n, D) -> (N_n)
            logits[mask.logical_not()] = -float('inf')

            all_logits.append(logits)

        return torch.cat(all_logits, dim=0), all_logits
    def get_location_orientation(
        self,
        spatial_embeds: torch.Tensor, # (sum M*N, D)
        graph_embeds: torch.Tensor, # (sum N_n, D)
        spatial_hw: torch.Tensor, # (B, 2)
        spatial_machine_mask: torch.Tensor, # (sum M*N, 4)
        node_lengths: torch.Tensor, # (B)
        selected: torch.Tensor, # (B)
        **kwargs
    ):
        start_idx = F.pad(node_lengths, (1, 0), value=0).cumsum(dim=0)[:-1]
        memb = graph_embeds[start_idx + selected]
        mloq = self.mlo_q_proj(memb) # (B, D)
        mlok = self.mlo_k_proj(spatial_embeds).reshape(-1, 4, self.config.intermediate_size) # (sum M*N, 4, D)

        slen = spatial_hw.prod(-1).tolist()
        all_logits = []
        for q, k, mask in zip(
            mloq,
            mlok.split(slen),
            spatial_machine_mask.split(slen)
        ):
            logits = torch.inner(q, k)
            logits[mask.logical_not()] = -float('inf')
            logits = logits.flatten()
            
            all_logits.append(logits)
        
        return torch.cat(all_logits, dim=0), all_logits
    
    def get_belt_location_orientation(
        self,
        context: torch.Tensor, # (B, D)
        spatial_embeds: torch.Tensor, # (sum M*N, D)
        spatial_belt_mask: torch.Tensor, # (sum M*N, 4)
        spatial_hw: torch.Tensor, # (B, 2)
        **kwargs
    ) -> torch.Tensor:
        bq = self.b_q_proj(context)
        bk = self.b_k_proj(spatial_embeds).reshape(-1, 4, self.config.intermediate_size)

        all_logits = []
        slen = spatial_hw.prod(-1).tolist()
        for q, k, mask in zip(
            bq,
            bk.split(slen),
            spatial_belt_mask.split(slen)
        ):
            logits = torch.inner(q, k)
            logits[mask.logical_not()] = -float('inf')
            logits = logits.flatten()

            all_logits.append(logits)

        return torch.cat(all_logits, dim=0), all_logits
    
    def get_power_location(
        self,
        context: torch.Tensor, # (B, D)
        spatial_embeds: torch.Tensor, # (sum M*N, D)
        spatial_hw: torch.Tensor, # (B, 2)
        spatial_power_mask: torch.Tensor, # (sum M*N)
        **kwargs
    ) -> torch.Tensor:
        pq = self.p_q_proj(context)
        pk = self.p_k_proj(spatial_embeds)

        all_logits = []
        slen = spatial_hw.prod(-1).tolist()
        for q, k, mask in zip(
            pq.unsqueeze(-2),
            pk.split(slen),
            spatial_power_mask.split(slen)
        ):
            logits = torch.inner(q, k)
            logits[mask.logical_not()] = -float('inf')

            all_logits.append(logits)

        return torch.cat(all_logits, dim=0), all_logits
