from dataclasses import dataclass, field


@dataclass
class EFLayoutSpatialConfig:
    type_num: int
    type_dim: int
    state_dim: int
    output_dim: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    dropout: float
    num_layers: int
    theta: float
    id_size: int = 0

@dataclass
class EFLayoutGraphConfig:
    type_num: int
    type_dim: int
    state_dim: int
    output_dim: int
    edge_dim: int
    hidden_size: int
    num_heads: int
    dropout: float
    num_layers: int
    id_size: int = 0

@dataclass
class EFLayoutConfig:
    spatial_config: EFLayoutSpatialConfig
    graph_config: EFLayoutGraphConfig
    id_buckets: int
    id_size: int
    hidden_size: int
    intermediate_size: int
    
    def __post_init__(self):
        self.spatial_config = EFLayoutSpatialConfig(**self.spatial_config)
        self.graph_config = EFLayoutGraphConfig(**self.graph_config)
        self.spatial_config.id_size = self.id_size
        self.graph_config.id_size = self.id_size
