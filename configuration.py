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

@dataclass
class EFLayoutConfig:
    spatial_config: EFLayoutSpatialConfig
    graph_config: EFLayoutGraphConfig
    hidden_size: int
    intermediate_size: int
