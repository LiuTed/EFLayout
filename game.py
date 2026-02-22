import random
import os
import yaml
from enum import Enum, auto
from dataclasses import dataclass
from typing import Sequence, Union, Optional, Any, Dict
from functools import lru_cache

import numpy as np
import torch


class Channels(Enum):
    MachineType = 0
    MachineId = auto()
    PortDirection = auto()
    ConnectedUp = auto()
    ConnectedRight = auto()
    # model features starts here
    Placed = auto()
    IsBelt = auto()
    IsMachine = auto()
    IsPower = auto()
    InputPort = auto()
    OutputPort = auto()
    PortUp = auto()
    PortRight = auto()
    Powered = auto()
    BeltRight = auto()
    BeltUp = auto()
    BeltCross = auto()
    BeltTurn = auto()
    Num = auto()

class Directions(Enum):
    Up = 0
    Right = auto()
    Down = auto()
    Left = auto()

@dataclass
class GraphNode:
    MachineType: int

@dataclass
class GraphEdge:
    Source: int
    Dest: int
    Weight: int
    Optional: bool = False

@dataclass
class Graph:
    Nodes: Sequence[GraphNode]
    Edges: Sequence[GraphEdge]

class NodeAttrs(Enum):
    Id = 0
    MachineType = auto()
    # model features starts here
    Placed = auto()
    Powered = auto()
    Source = auto()
    Sink = auto()
    Num = auto()

class EdgeAttrs(Enum):
    Abstract = 0
    Weight = auto()
    Connected = auto()
    Optional = auto()
    Num = auto()


RevDirections = [Directions.Down, Directions.Left, Directions.Up, Directions.Right]
DirectionDxDy = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DefaultInputs = {
    (3, 3): ((2, 0), (2, 1), (2, 2)),
    (5, 5): ((4, 0), (4, 1), (4, 2), (4, 3), (4, 4)),
    (3, 6): ((2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)),
}
DefaultOutputs = {
    (3, 3): ((0, 0), (0, 1), (0, 2)),
    (5, 5): ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    (3, 6): ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
}

with open(os.path.join(os.path.dirname(__file__), 'game_config.yaml'), 'r') as f:
    _config = yaml.safe_load(f)
    Machines = _config['machines']
    NameToMachine = {}
    for i, m in enumerate(Machines):
        if 'inputs' not in m:
            inputs = DefaultInputs[tuple(m['size'])]
        else:
            inputs = m['inputs']
        inputs = ((*inp, 0) if len(inp) == 2 else inp for inp in inputs)
        m['inputs'] = tuple(inputs)
        if 'outputs' not in m:
            outputs = DefaultOutputs[tuple(m['size'])]
        else:
            outputs = m['outputs']
        outputs = ((*out, 0) if len(out) == 2 else out for out in outputs)
        m['outputs'] = tuple(outputs)
        NameToMachine[m['EN']] = i
    PowerSize = _config['power']['size']
    PowerRange = _config['power']['range']
    Rewards = _config['rewards']
    Levels = _config['levels']


class EndField:
    def __init__(self, h: int, w: int, expect_graph: Graph, depot_edges: Sequence[int] = [0]):
        self.h = h
        self.w = w
        self.minx = h
        self.miny = w
        self.maxx = -1
        self.maxy = -1
        self.id_offset = random.randint(0, 32768)
        self.depot_edges = depot_edges

        self.field = np.zeros((h, w, Channels.Num.value), dtype=np.int32) # Left-Up = 0, 0
        self.field[..., Channels.ConnectedUp.value] = -1
        self.field[..., Channels.ConnectedRight.value] = -1
        
        self.nodes = np.zeros((len(expect_graph.Nodes) + 2, NodeAttrs.Num.value), dtype=np.int32) # Add source and sink
        self.nodes[..., NodeAttrs.Id.value] = np.arange(len(expect_graph.Nodes) + 2) + self.id_offset

        source_sink_edges = []
        sourceid = len(expect_graph.Nodes)
        sinkid = len(expect_graph.Nodes) + 1
        for i, t in enumerate(expect_graph.Nodes):
            self.nodes[i, NodeAttrs.MachineType.value] = t.MachineType
            if t.MachineType in [NameToMachine['Depot-Unloader'], NameToMachine['Core']]:
                source_sink_edges.append((sourceid, i))
            elif t.MachineType in [NameToMachine['Depot-Loader'], NameToMachine['Core'], NameToMachine['Protocol-Stash']]:
                source_sink_edges.append((i, sinkid))
        self.nodes[sourceid:sinkid+1, NodeAttrs.MachineType.value] = NameToMachine['AbstractNode']
        self.nodes[sourceid:sinkid+1, NodeAttrs.Placed.value] = 1
        self.nodes[sourceid:sinkid+1, NodeAttrs.Powered.value] = 1
        self.nodes[sourceid, NodeAttrs.Source.value] = 1
        self.nodes[sinkid, NodeAttrs.Sink.value] = 1

        edges = [(e.Source, e.Dest) for e in expect_graph.Edges]
        self.src_to_dst = {e.Source: [] for e in expect_graph.Edges}
        self.dst_to_src = {e.Dest: [] for e in expect_graph.Edges}
        self.edges = np.array(edges + source_sink_edges, dtype=np.int32)
        self.edge_attrs = np.zeros((len(edges) + len(source_sink_edges), EdgeAttrs.Num.value), dtype=np.int32)
        self.edge_attrs[len(edges):, EdgeAttrs.Abstract.value] = 1
        self.edge_attrs[len(edges):, EdgeAttrs.Connected.value] = 1
        for i, e in enumerate(expect_graph.Edges):
            self.src_to_dst[e.Source].append(e.Dest)
            self.dst_to_src[e.Dest].append(e.Source)
            if e.Optional:
                self.edge_attrs[i, EdgeAttrs.Optional.value] = 1
            self.edge_attrs[i, EdgeAttrs.Weight.value] = e.Weight
        self.edge_to_idx = {tuple(e): i for i, e in enumerate(edges)}

        self.global_mid = 1
        self._finished = False
        self._finish_reason = ''

    @property
    def finished(self) -> bool:
        return self._finished

    def get_action_mask(self) -> torch.Tensor:
        mask = torch.zeros(3, dtype=torch.bool)
        mask[0] = self.get_machine_mask().any()
        mask[1] = self.get_spatial_belt_mask().any()
        mask[2] = self.get_spatial_power_mask().any()
        return mask

    def get_machine_mask(self) -> torch.Tensor:
        return torch.tensor(self.nodes[..., NodeAttrs.Placed.value] == 0, dtype=torch.bool).flatten()
    
    # @lru_cache()
    def get_spatial_machine_mask(self, machine_id) -> torch.Tensor:
        mtype = self.nodes[machine_id, NodeAttrs.MachineType.value]
        r, c = Machines[mtype]['size']
        depot = Machines[mtype].get('depot', False)

        mask = np.zeros((self.h, self.w, 2), dtype=np.bool)
        prefix = np.zeros((self.h+1, self.w+1), dtype=np.int32)
        prefix[1:, 1:] = np.cumsum(np.cumsum(self.field[..., Channels.Placed.value], axis=0), axis=1)
        sum_mask = prefix[:self.h-r+1, :self.w-c+1] + prefix[r:, c:] - prefix[r:, :self.w-c+1] - prefix[:self.h-r+1, c:]
        mask[:self.h-r+1, :self.w-c+1, 0] = sum_mask == 0

        if r != c:
            r, c = c, r
            prefix = np.zeros((self.h+1, self.w+1), dtype=np.int32)
            prefix[1:, 1:] = np.cumsum(np.cumsum(self.field[..., Channels.Placed.value], axis=0), axis=1)
            sum_mask = prefix[:self.h-r+1, :self.w-c+1] + prefix[r:, c:] - prefix[r:, :self.w-c+1] - prefix[:self.h-r+1, c:]
            mask[:self.h-r+1, :self.w-c+1, 1] = sum_mask == 0
        else:
            mask[..., 1] = mask[..., 0]
        
        mask = np.concat([mask, mask], axis=-1)
        
        if depot:
            loader = Machines[mtype]['EN'] == 'Depot-Loader'
            depot_mask = np.zeros((self.h, self.w, 4), dtype=np.bool)
            for depot_edge in self.depot_edges:
                for d in range(4):
                    r, c = Machines[mtype]['size']
                    if d in [1, 3]:
                        r, c = c, r
                    if depot_edge == 0:
                        depot_mask[0, :, d] = (d == (2 if not loader else 0))
                    elif depot_edge == 1:
                        depot_mask[:, -c, d] = (d == (3 if not loader else 2))
                    elif depot_edge == 2:
                        depot_mask[-r, :, d] = (d == (0 if not loader else 2))
                    elif depot_edge == 3:
                        depot_mask[:, 0, d] = (d == (1 if not loader else 3))
        else:
            depot_mask = np.ones((self.h, self.w, 4), dtype=np.bool)
        
        mask = np.logical_and(mask, depot_mask)
        return torch.from_numpy(mask).contiguous().to(dtype=torch.bool).reshape(-1, 4)
    
    # @lru_cache()
    def get_spatial_belt_mask(self) -> torch.Tensor:
        io_mask = np.logical_and(
            np.logical_or(
                self.field[..., Channels.InputPort.value] == 1,
                self.field[..., Channels.OutputPort.value] == 1
            ),
            self.field[..., Channels.IsBelt.value] == 0
        )
        io_mask = np.expand_dims(io_mask, axis=-1)
        directions = np.expand_dims(np.arange(4), axis=(0, 1))
        io_dir_mask = self.field[..., Channels.PortDirection.value, np.newaxis] == directions
        io_mask = np.logical_and(io_mask, io_dir_mask)
        io_mask = torch.from_numpy(io_mask).to(dtype=torch.bool)
        for x in range(self.h):
            for y in range(self.w):
                if self.field[x, y, Channels.InputPort.value] == 1 :
                    d = self.field[x, y, Channels.PortDirection.value]
                    xx = x + DirectionDxDy[RevDirections[d].value][0]
                    yy = y + DirectionDxDy[RevDirections[d].value][1]
                elif self.field[x, y, Channels.OutputPort.value] == 1:
                    d = self.field[x, y, Channels.PortDirection.value]
                    xx = x + DirectionDxDy[d][0]
                    yy = y + DirectionDxDy[d][1]
                else:
                    continue
                if xx < 0 or xx >= self.h or yy < 0 or yy >= self.w:
                    io_mask[x, y] = False
                elif self.field[xx, yy, Channels.IsMachine.value] == 1 or self.field[xx, yy, Channels.IsPower.value] == 1:
                    io_mask[x, y] = False
        
        cross_mask = np.logical_and(
            self.field[..., Channels.IsBelt.value] == 1,
            self.field[..., Channels.IsMachine.value] == 0
        )
        cross_mask = np.logical_and(
            cross_mask,
            self.field[..., Channels.BeltCross.value] == 0
        )
        cross_mask = np.logical_and(
            cross_mask,
            self.field[..., Channels.BeltTurn.value] == 0
        )
        cross_right_mask = np.logical_and(
            cross_mask,
            self.field[..., Channels.BeltRight.value] == 0
        )
        cross_up_mask = np.logical_and(
            cross_mask,
            self.field[..., Channels.BeltUp.value] == 0
        )
        cross_mask = np.stack([cross_up_mask, cross_right_mask, cross_up_mask, cross_right_mask], axis=-1)
        cross_mask = torch.from_numpy(cross_mask).to(dtype=torch.bool)

        place_mask = self.field[..., Channels.Placed.value] == 0
        place_mask = torch.from_numpy(place_mask).to(torch.bool).unsqueeze(-1).expand_as(io_mask)
        
        return (io_mask | cross_mask | place_mask).reshape(-1, 4)

    # @lru_cache()
    def get_spatial_power_mask(self) -> torch.Tensor:
        r, c = PowerSize
        mask = np.zeros((self.h, self.w), dtype=np.bool)
        prefix = np.zeros((self.h+1, self.w+1), dtype=np.int32)
        prefix[1:, 1:] = np.cumsum(np.cumsum(self.field[..., Channels.Placed.value], axis=0), axis=1)
        sum_mask = prefix[:self.h-r+1, :self.w-c+1] + prefix[r:, c:] - prefix[r:, :self.w-c+1] - prefix[:self.h-r+1, c:]
        mask[:self.h-r+1, :self.w-c+1] = sum_mask == 0
        return torch.from_numpy(mask).contiguous().to(dtype=torch.bool).flatten()
    
    def clean_cache(self):
        # self.get_spatial_belt_mask.cache_clear()
        # self.get_spatial_machine_mask.cache_clear()
        # self.get_spatial_power_mask.cache_clear()
        pass

    def get_states(self) -> Dict[str, Any]:
        spatial_states = torch.tensor(self.field[..., Channels.Placed.value:]).to(dtype=torch.float32).flatten(0, 1)
        spatial_types = torch.tensor(self.field[..., Channels.MachineType.value]).flatten(0, 1)
        spatial_ids = torch.tensor(self.field[..., Channels.MachineId.value]).to(dtype=torch.int32).flatten(0, 1) + self.id_offset
        graph_node_states = torch.tensor(self.nodes[..., NodeAttrs.Placed.value:]).to(dtype=torch.float32)
        graph_node_types = torch.tensor(self.nodes[..., NodeAttrs.MachineType.value])
        graph_node_ids = torch.tensor(self.nodes[..., NodeAttrs.Id.value]).to(dtype=torch.int32)
        return {
            'spatial_info': {
                'states': spatial_states,
                'types': spatial_types,
                'ids': spatial_ids,
            },
            'spatial_hw': torch.tensor(((self.h, self.w),), dtype=torch.int32),
            'graph_nodes': {
                'states': graph_node_states,
                'types': graph_node_types,
                'ids': graph_node_ids,
            },
            'edge_index': torch.tensor(self.edges).transpose(0, 1),
            'edge_attrs': torch.tensor(self.edge_attrs),
            'node_lengths': torch.tensor((self.nodes.shape[0],)),
        }
    
    def check_all_connected(self) -> bool:
        for dst, srcs in self.dst_to_src.items():
            required = True
            optional = False
            for src in srcs:
                idx = self.edge_to_idx[(src, dst)]
                if self.edge_attrs[idx, EdgeAttrs.Optional.value] == 0:
                    if self.edge_attrs[idx, EdgeAttrs.Connected.value] < self.edge_attrs[idx, EdgeAttrs.Weight.value]:
                        required = False
                else:
                    if self.edge_attrs[idx, EdgeAttrs.Connected.value] >= self.edge_attrs[idx, EdgeAttrs.Weight.value]:
                        optional = True
            if not (required and optional):
                return False
        return True
    
    def check_finished(self) -> bool:
        if not np.all(self.nodes[..., NodeAttrs.Placed.value] == 1):
            return False
        if not np.all(self.nodes[..., NodeAttrs.Powered.value] == 1):
            return False
        if not self.check_all_connected():
            return False
        
        return True
    
    def check_dead(self) -> bool:
        if not np.all(self.nodes[..., NodeAttrs.Placed.value] == 1):
            for i in range(self.nodes.shape[0]):
                if self.nodes[i, NodeAttrs.Placed.value] == 1:
                    continue
                if not self.get_spatial_machine_mask(i).any():
                    self._finish_reason = f"Machine {i} is not connected to any other machines, mask={self.get_spatial_machine_mask(i)}"
                    return True
        if not np.all(self.nodes[..., NodeAttrs.Powered.value] == 1):
            if not self.get_spatial_power_mask().any():
                self._finish_reason = f"No more Power source can be placed, mask={self.get_spatial_power_mask()}"
                return True
        if not self.get_spatial_belt_mask().any():
            self._finish_reason = f"No more Belt can be placed, mask={self.get_spatial_belt_mask()}"
            return True
        return False

    def put_machine_model(self, machine_id, idx):
        pos, direction = idx // 4, idx % 4
        return self.put_machine(machine_id, (pos // self.w, pos % self.w), direction)

    def put_machine(self, machine_id, position, direction):
        machine_type = self.nodes[machine_id, NodeAttrs.MachineType.value]
        r, c = Machines[machine_type]['size']
        inputs = Machines[machine_type]['inputs']
        outputs = Machines[machine_type]['outputs']
        reward = 0
        if direction in [Directions.Right.value, Directions.Left.value]:
            r, c = c, r
        x, y = position
        if x + r > self.h or y + c > self.w:
            raise RuntimeError()
        for i in range(x, x + r):
            for j in range(y, y + c):
                if self.field[i, j, Channels.Placed.value] == 1:
                    raise RuntimeError(f'conflict at ({i}, {j}), input=({machine_type}, {x}, {y}, {direction}), self=\n{str(self)}\nmask={self.get_spatial_machine_mask(machine_id).reshape(self.h, self.w, 4)}')
        for i in range(x, x + r):
            for j in range(y, y + c):
                self.field[i, j, Channels.Placed.value] = 1
                self.field[i, j, Channels.IsMachine.value] = 1
                self.field[i, j, Channels.MachineType.value] = machine_type
                self.field[i, j, Channels.MachineId.value] = machine_id
                if self.field[i, j, Channels.Powered.value] == 1:
                    if self.nodes[machine_id, NodeAttrs.Powered.value] != 1:
                        reward += Rewards['machine_powered']
                    self.nodes[machine_id, NodeAttrs.Powered.value] = 1

        # rotate clockwise 90 first
        if direction in [Directions.Right.value, Directions.Left.value]:
            inputs = ((yy, c - xx - 1, (d+1)%4) for xx, yy, d in inputs)
            outputs = ((yy, c - xx - 1, (d+1)%4) for xx, yy, d in outputs)
        # rotate 180 then
        if direction in [Directions.Down.value, Directions.Left.value]:
            inputs = ((r - xx - 1, c - yy - 1, (d+2)%4) for xx, yy, d in inputs)
            outputs = ((r - xx - 1, c - yy - 1, (d+2)%4) for xx, yy, d in outputs)
        for xx, yy, pd in inputs:
            self.field[x + xx, y + yy, Channels.InputPort.value] = 1
            self.field[x + xx, y + yy, Channels.PortDirection.value] = pd
            if pd in [Directions.Up.value, Directions.Down.value]:
                self.field[x + xx, y + yy, Channels.PortUp.value] = 1 if pd == Directions.Up.value else -1
            else:
                self.field[x + xx, y + yy, Channels.PortRight.value] = 1 if pd == Directions.Right.value else -1
        for xx, yy, pd in outputs:
            self.field[x + xx, y + yy, Channels.OutputPort.value] = 1
            self.field[x + xx, y + yy, Channels.PortDirection.value] = pd
            if pd in [Directions.Up.value, Directions.Down.value]:
                self.field[x + xx, y + yy, Channels.PortUp.value] = 1 if pd == Directions.Up.value else -1
            else:
                self.field[x + xx, y + yy, Channels.PortRight.value] = 1 if pd == Directions.Right.value else -1

        self.nodes[machine_id, NodeAttrs.Placed.value] = 1
        all_placed = self.nodes[:-2, NodeAttrs.Placed.value].all()
        
        self.minx = min(self.minx, x)
        self.miny = min(self.miny, y)
        self.maxx = max(self.maxx, x + r - 1)
        self.maxy = max(self.maxy, y + c - 1)
        if all_placed:
            reward += Rewards['all_machine_placed']
        if self.check_finished():
            reward += Rewards['finished'] * (1. - (self.maxx - self.minx + 1) * (self.maxy - self.miny + 1) / self.h / self.w)
            self._finished = True
        elif self.check_dead():
            reward += Rewards['dead_end']
            self._finished = True
        self.clean_cache()
        return reward
    
    def _broadcast_connection(self, *position, direction=None):
        x, y = position
        assert self.field[x, y, Channels.IsBelt.value] == 1
        if direction is None:
            assert self.field[x, y, Channels.BeltCross.value] == 0
            if self.field[x, y, Channels.BeltUp.value] != 0:
                direction = Directions.Up.value if self.field[x, y, Channels.BeltUp.value] == 1 else Directions.Down.value
            elif self.field[x, y, Channels.BeltRight.value] != 0:
                direction = Directions.Right.value if self.field[x, y, Channels.BeltRight.value] == 1 else Directions.Left.value
        
        while True:
            if direction in [Directions.Up.value, Directions.Down.value]:
                if self.field[x, y, Channels.BeltUp.value] == 0:
                    return 0
                conn = self.field[x, y, Channels.ConnectedUp.value]
            else:
                if self.field[x, y, Channels.BeltRight.value] == 0:
                    return 0
                conn = self.field[x, y, Channels.ConnectedRight.value]
            if conn == -1:
                return 0
            xx = x + DirectionDxDy[direction][0]
            yy = y + DirectionDxDy[direction][1]
            
            if xx < 0 or xx >= self.h or yy < 0 or yy >= self.w:
                return 0
            
            if self.field[xx, yy, Channels.IsBelt.value] == 0:
                return 0
            if self.field[xx, yy, Channels.BeltCross.value] != 0:
                chann = Channels.ConnectedUp.value if direction in [Directions.Up.value, Directions.Down.value] else Channels.ConnectedRight.value
                self.field[xx, yy, chann] =  conn
                dd = direction
            else:
                chann = Channels.ConnectedRight.value if self.field[xx, yy, Channels.BeltUp.value] != 0 else Channels.ConnectedUp.value
                self.field[xx, yy, chann] = conn
                if self.field[xx, yy, Channels.BeltUp.value] != 0:
                    dd = Directions.Up.value if self.field[xx, yy, Channels.BeltUp.value] == 1 else Directions.Down.value
                else:
                    dd = Directions.Right.value if self.field[xx, yy, Channels.BeltRight.value] == 1 else Directions.Left.value
            
            if self.field[xx, yy, Channels.IsMachine.value] == 1:
                dst = self.field[xx, yy, Channels.MachineId.value]
                if (conn, dst) in self.edge_to_idx:
                    idx = self.edge_to_idx[(conn, dst)]
                    self.edge_attrs[idx, EdgeAttrs.Connected.value] += 1
                    if self.check_all_connected():
                        return Rewards['all_connected']
                    return Rewards['connected']
                else:
                    return Rewards['unmatched']
            
            x, y, direction = xx, yy, dd
    
    def put_belt_model(self, idx):
        pos, direction = idx // 4, idx % 4
        return self.put_belt((pos // self.w, pos % self.w), direction)
    
    def put_belt(self, position, direction):
        updown = direction in [Directions.Up.value, Directions.Down.value]
        x, y = position
        reward = 0
        if self.field[x, y, Channels.Placed.value] == 0:
            self.field[x, y, Channels.Placed.value] = 1
            self.field[x, y, Channels.IsBelt.value] = 1
            if direction in [Directions.Right.value, Directions.Left.value]:
                self.field[x, y, Channels.BeltRight.value] = 1 if direction == Directions.Right.value else -1
            elif direction in [Directions.Up.value, Directions.Down.value]:
                self.field[x, y, Channels.BeltUp.value] = 1 if direction == Directions.Up.value else -1

            hasStraight = False
            hasTurn = False
            straightFrom = -1
            turnFrom = -1
            for d in range(4):
                if d == direction:
                    continue
                xx = x + DirectionDxDy[d][0]
                yy = y + DirectionDxDy[d][1]
                if xx < 0 or xx >= self.h or yy < 0 or yy >= self.w:
                    continue
                if self.field[xx, yy, Channels.IsBelt.value] != 1:
                    continue
                pointsTo = False
                dd = RevDirections[d]
                if dd in [Directions.Right, Directions.Left]:
                    if self.field[xx, yy, Channels.BeltRight.value] == (1 if dd == Directions.Right else -1):
                        pointsTo = True
                elif dd in [Directions.Up, Directions.Down]:
                    if self.field[xx, yy, Channels.BeltUp.value] == (1 if dd == Directions.Up else -1):
                        pointsTo = True
                ddupdown = dd in [Directions.Up, Directions.Down]
                if pointsTo:
                    if dd == direction:
                        hasStraight = True
                        straightFrom = self.field[
                            xx, yy,
                            Channels.ConnectedUp.value if ddupdown else Channels.ConnectedRight.value
                        ]
                    else:
                        hasTurn = True
                        turnFrom = self.field[
                            xx, yy,
                            Channels.ConnectedUp.value if ddupdown else Channels.ConnectedRight.value
                        ]
            if not hasStraight and hasTurn:
                self.field[x, y, Channels.BeltTurn.value] = 1
                self.field[x, y, Channels.ConnectedUp.value if updown else Channels.ConnectedRight.value] = turnFrom
                reward += Rewards['belt_extend']
                reward += self._broadcast_connection(x, y)
            elif hasStraight:
                self.field[x, y, Channels.ConnectedUp.value if updown else Channels.ConnectedRight.value] = straightFrom
                reward += Rewards['belt_extend']
                reward += self._broadcast_connection(x, y)
            else:
                reward += Rewards['alone_belt']

        elif self.field[x, y, Channels.InputPort.value] == 1 or self.field[x, y, Channels.OutputPort.value] == 1:
            if self.field[x, y, Channels.IsBelt.value]:
                raise RuntimeError()
            self.field[x, y, Channels.IsBelt.value] = 1
            if direction !=  self.field[x, y, Channels.PortDirection.value]:
                raise RuntimeError()
            if not updown:
                self.field[x, y, Channels.BeltRight.value] = 1 if direction == Directions.Right.value else -1
            else:
                self.field[x, y, Channels.BeltUp.value] = 1 if direction == Directions.Up.value else -1
            
            if self.field[x, y, Channels.InputPort.value] == 1:
                self.field[x, y, Channels.ConnectedUp.value if updown else Channels.ConnectedRight.value] = self.field[x, y, Channels.MachineId.value]
                reward += Rewards['belt_extend']
                reward += self._broadcast_connection(x, y)
            else:
                xx = x + DirectionDxDy[RevDirections[direction].value][0]
                yy = y + DirectionDxDy[RevDirections[direction].value][1]
                if self.field[xx, yy, Channels.IsBelt.value] == 1:
                    reward += self._broadcast_connection(xx, yy, direction=direction)
                else:
                    reward += Rewards['alone_belt']
        
        elif self.field[x, y, Channels.IsBelt.value] == 1:
            if direction in [Directions.Right.value, Directions.Left.value]:
                if self.field[x, y, Channels.BeltRight.value] != 0:
                    raise RuntimeError()
                self.field[x, y, Channels.BeltUp.value] = 1 if direction == Directions.Right.value else -1
                self.field[x, y, Channels.BeltCross.value] = 1
            elif direction in [Directions.Up.value, Directions.Down.value]:
                if self.field[x, y, Channels.BeltUp.value] != 0:
                    raise RuntimeError()
                self.field[x, y, Channels.BeltUp.value] = 1 if direction == Directions.Up.value else -1
                self.field[x, y, Channels.BeltCross.value] = 1
            
            xx = x + DirectionDxDy[RevDirections[direction].value][0]
            yy = y + DirectionDxDy[RevDirections[direction].value][1]
            if xx < 0 or xx >= self.h or yy < 0 or yy >= self.w:
                reward += Rewards['alone_belt']
            elif self.field[xx, yy, Channels.IsBelt.value] == 1:
                reward += self._broadcast_connection(xx, yy, direction=direction)
            else:
                reward += Rewards['alone_belt']
        
        else:
            raise RuntimeError()

        self.minx = min(self.minx, x)
        self.miny = min(self.miny, y)
        self.maxx = max(self.maxx, x)
        self.maxy = max(self.maxy, y)
        
        if self.check_finished():
            reward += Rewards['finished'] * (1. - (self.maxx - self.minx + 1) * (self.maxy - self.miny + 1) / self.h / self.w)
            self._finished = True
        elif self.check_dead():
            reward += Rewards['dead_end']
            self._finished = True
        self.clean_cache()
        return reward

    def put_power_model(self, pos):
        return self.put_power((pos // self.w, pos % self.w))

    def put_power(self, position):
        x, y = position
        reward = 0
        for i in range(x, x+PowerSize[0]):
            if i >= self.h:
                raise RuntimeError()
            for j in range(y, y+PowerSize[1]):
                if j >= self.w:
                    raise RuntimeError()
                if self.field[i, j, Channels.Placed.value] == 1:
                    raise RuntimeError(f'conflict at ({i}, {j}), input=({x}, {y}), self={str(self)}')
        
        for i in range(x, x+PowerSize[0]):
            for j in range(y, y+PowerSize[1]):
                self.field[i, j, Channels.Placed.value] = 1
                self.field[i, j, Channels.IsPower.value] = 1

        for i in range(x-PowerRange[0], x+PowerRange[1]):
            if i < 0 or i >= self.h:
                continue
            for j in range(y-PowerRange[0], y+PowerRange[1]):
                if j < 0 or j >= self.w:
                    continue
                self.field[i, j, Channels.Powered.value] = 1
                if self.field[i, j, Channels.IsMachine.value] == 1:
                    machine_id = self.field[i, j, Channels.MachineId.value]
                    if not self.nodes[machine_id, NodeAttrs.Powered.value]:
                        reward += Rewards['machine_powered']
                    self.nodes[machine_id, NodeAttrs.Powered.value] = 1
                    
        self.minx = min(self.minx, x)
        self.miny = min(self.miny, y)
        self.maxx = max(self.maxx, x + PowerSize[0] - 1)
        self.maxy = max(self.maxy, y + PowerSize[1] - 1)
        if self.check_finished():
            reward += Rewards['finished'] * (1. - (self.maxx - self.minx + 1) * (self.maxy - self.miny + 1) / self.h / self.w)
            self._finished = True
        elif self.check_dead():
            reward += Rewards['dead_end']
            self._finished = True
        self.clean_cache()
        return reward
        
    def render(self, block=True, file=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        plt.figure(figsize=(10, 10))
        plt.xlim(0, self.w)
        plt.ylim(0, self.h)
        plt.gca().invert_yaxis()
        plt.grid(True, color='gray', linewidth=0.5)
        
        for i in range(self.h):
            for j in range(self.w):
                if self.field[i, j, Channels.IsMachine.value] == 1:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='none',
                                             facecolor='black' if self.field[i, j, Channels.Powered.value] == 0 else 'midnightblue')
                    plt.gca().add_patch(rect)
                    if self.field[i, j, Channels.InputPort.value] == 1:
                        if self.field[i, j, Channels.PortUp.value] == 1:
                            plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, 0, -0.5, color='wheat'))
                        if self.field[i, j, Channels.PortUp.value] == -1:
                            plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, 0, 0.5, color='wheat'))
                        if self.field[i, j, Channels.PortRight.value] == 1:
                            plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, 0.5, 0, color='wheat'))
                        if self.field[i, j, Channels.PortRight.value] == -1:
                            plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, -0.5, 0, color='wheat'))
                    if self.field[i, j, Channels.OutputPort.value] == 1:
                        if self.field[i, j, Channels.PortUp.value] == 1:
                            plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, 0, -0.5, color='lightgreen'))
                        if self.field[i, j, Channels.PortUp.value] == -1:
                            plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, 0, 0.5, color='lightgreen'))
                        if self.field[i, j, Channels.PortRight.value] == 1:
                            plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, 0.5, 0, color='lightgreen'))
                        if self.field[i, j, Channels.PortRight.value] == -1:
                            plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, -0.5, 0, color='lightgreen'))
                elif self.field[i, j, Channels.IsBelt.value] == 1:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='none',
                                             facecolor='darkorange')
                    plt.gca().add_patch(rect)
                    if self.field[i, j, Channels.BeltUp.value] == 1:
                        plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, 0, -0.5, color='gold'))
                    if self.field[i, j, Channels.BeltUp.value] == -1:
                        plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, 0, 0.5, color='gold'))
                    if self.field[i, j, Channels.BeltRight.value] == 1:
                        plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, 0.5, 0, color='gold'))
                    if self.field[i, j, Channels.BeltRight.value] == -1:
                        plt.gca().add_patch(patches.Arrow(j+0.5, i+0.5, -0.5, 0, color='gold'))
                elif self.field[i, j, Channels.IsPower.value] == 1:
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='none',
                                             facecolor='blue')
                    plt.gca().add_patch(rect)
        if file is not None:
            plt.savefig(file)
            plt.close()
        else:
            plt.show(block=block)

    def __str__(self):
        output = ''
        for i in range(self.h):
            for j in range(self.w):
                if self.field[i, j, Channels.IsMachine.value] == 1:
                    if self.field[i, j, Channels.InputPort.value] == 1:
                        output += 'I'
                    elif self.field[i, j, Channels.OutputPort.value] == 1:
                        output += 'O'
                    else:
                        output += 'M'
                elif self.field[i, j, Channels.IsBelt.value] == 1:
                    if self.field[i, j, Channels.BeltUp.value] == 1:
                        output += '^'
                    if self.field[i, j, Channels.BeltUp.value] == -1:
                        output += 'v'
                    if self.field[i, j, Channels.BeltRight.value] == 1:
                        output += '>'
                    if self.field[i, j, Channels.BeltRight.value] == -1:
                        output += '<'
                elif self.field[i, j, Channels.IsPower.value] == 1:
                    output += 'P'
                else:
                    output += '.'
            output += '\n'
        for node in self.nodes:
            if node[NodeAttrs.Placed.value] == 1:
                if node[NodeAttrs.Source.value] == 1:
                    output += 'O'
                elif node[NodeAttrs.Sink.value] == 1:
                    output += 'I'
                elif node[NodeAttrs.Powered.value] == 1:
                    output += 'D'
                else:
                    output += 'P'
            else:
                output += 'N'
        output += f'\n{self._finished=}, {self._finish_reason=}'
        return output

def get_max_level() -> int:
    return len(Levels) - 1

def generate_new_game(level: int) -> EndField:
    if level >= len(Levels):
        level = len(Levels) - 1
    hrange = Levels[level]['h']
    wrange = Levels[level]['w']
    nrange = Levels[level]['num_machines']
    idrange = Levels[level]['machine_range']
    outportrange = Levels[level]['max_output_ports']
    core_prob = Levels[level]['core_prob']
    indegreerange = Levels[level]['input_degrees']
    depot_edges = Levels[level]['depot_edges']
    
    h = random.randint(hrange[0], hrange[1])
    w = random.randint(wrange[0], wrange[1])
    n = random.randint(nrange[0], nrange[1])
    outports = random.randint(1, min(outportrange, n))
    hascore = random.random() < core_prob
    nodes = []
    edges = []
    for i in range(n):
        id = random.randint(idrange[0], idrange[1])
        nodes.append(GraphNode(id))
    if hascore:
        nodes.append(GraphNode(NameToMachine['Core']))
        core_id = len(nodes)-1
    for i in range(outports):
        nodes.append(GraphNode(NameToMachine['Depot-Unloader']))
        edges.append(GraphEdge(len(nodes)-1, i, 1, Optional=hascore))
        if hascore:
            edges.append(GraphEdge(core_id, i, 1, Optional=True))
    for i in range(outports, n):
        if i - outports < outports:
            edges.append(GraphEdge(i - outports, i, 1))
        else:
            for j in range(random.randint(indegreerange[0], indegreerange[1])):
                edges.append(GraphEdge(random.randint(outports, i-1), i, random.randint(1, 2)))
    
    if not hascore:
        for i in range(random.randint(indegreerange[0], indegreerange[1])):
            nodes.append(GraphNode(NameToMachine['Depot-Loader']))
            edges.append(GraphEdge(n-i-1, len(nodes)-1, 1))
    else:
        for i in range(random.randint(indegreerange[0], indegreerange[1])):
            edges.append(GraphEdge(n-i-1, core_id, 1))

    return EndField(h, w, Graph(nodes, edges), depot_edges)

if __name__ == '__main__':
    game = generate_new_game(0)
    game.put_power((0, 0))
    game.put_machine(0, (0, 2), 2)
    game.put_belt((3, 3), 2)
    game.put_belt((4, 3), 1)
    game.put_belt((4, 3), 2)
    print(game.get_spatial_power_mask().reshape(game.h, game.w))
    game.render()
