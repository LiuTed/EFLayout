from enum import Enum, auto
from dataclasses import dataclass
from typing import Sequence, Union, Optional, Any, Dict

import numpy as np
import torch


class Channels(Enum):
    MachineType = auto()
    MachineId = auto()
    Placed = auto()
    IsBelt = auto()
    IsMachine = auto()
    IsPower = auto()
    MachineDirection = auto()
    InputPort = auto()
    OutputPort = auto()
    Powered = auto()
    BeltRight = auto()
    BeltUp = auto()
    BeltCross = auto()
    BeltTurn = auto()
    Num = auto()

class Directions(Enum):
    Up = auto()
    Right = auto()
    Down = auto()
    Left = auto()

@dataclass
class Graph:
    Nodes: Sequence[int]
    Edges: Sequence[Sequence[int]]

class NodeAttrs(Enum):
    MachineType = auto()
    Placed = auto()
    Powered = auto()
    Num = auto()

RevDirections = [Directions.Down, Directions.Left, Directions.Up, Directions.Right]
DirectionDxDy = [(-1, 0), (0, 1), (1, 0), (0, -1)]

Default33Inputs = ((2, 0), (2, 1), (2, 2))
Default33Outputs = ((0, 0), (0, 1), (0, 2))
Default35Inputs = ((2, 0), (2, 1), (2, 2), (2, 3), (2, 4))
Default35Outputs = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
Default55Inputs = ((4, 0), (4, 1), (4, 2), (4, 3), (4, 4))
Default55Outputs = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
Default36Inputs = ((2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5))
Default36Outputs = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
Machines = [
    ('Jinglian', 3, 3, Default33Inputs, Default33Outputs),
    ('Fensui', 3, 3, Default33Inputs, Default33Outputs),
    ('Fengzhuang', 3, 5, Default35Inputs, Default35Outputs),
    ('Zhongzhi', 5, 5, Default55Inputs, Default55Outputs),
    ('Caizhong', 5, 5, Default55Inputs, Default55Outputs),
]


class EndField:
    def __init__(self, h: int, w: int, expect_graph: Graph):
        self.h = h
        self.w = w
        self.field = np.zeros((h, w, Channels.Num.value), dtype=np.int32) # Left-Up = 0, 0
        self.nodes = np.zeros((len(expect_graph.Nodes), NodeAttrs.Num.value), dtype=np.int8)
        for i, t in enumerate(expect_graph.Nodes):
            self.nodes[i, NodeAttrs.MachineType.value] = t
        self.edges = np.array(expect_graph.Edges, dtype=np.int32)
        self.global_mid = 1

    def get_action_mask(self) -> torch.Tensor:
        mask = torch.zeros(3, dtype=torch.bool)
        mask[0] = self.get_machine_mask().any()
        mask[1] = self.get_spatial_belt_mask().any()
        mask[2] = self.get_spatial_power_mask().any()
        return mask

    def get_machine_mask(self) -> torch.Tensor:
        return torch.tensor(self.nodes[..., NodeAttrs.Placed.value] == 0, dtype=torch.bool)
    
    def get_spatial_machine_mask(self, machine_id) -> torch.Tensor:
        mtype = self.nodes[machine_id, NodeAttrs.MachineType.value]
        r, c = Machines[mtype][1:3]
        mask = np.zeros((self.h, self.w, 2), dtype=np.bool)
        if r == 1:
            rowmask = self.field[..., Channels.Placed.value]
        else:
            rowcum = self.field[..., Channels.Placed.value].cumsum(axis=0)
            rowmask = rowcum[:-r+1] == rowcum[r-1:]
            rowmask = np.logical_and(rowmask, self.field[:-r+1, :, Channels.Placed.value])
        if c == 1:
            colmask = self.field[..., Channels.Placed.value]
        else:
            colcum = self.field[..., Channels.Placed.value].cumsum(axis=1)
            colmask = colcum[:, :-c+1] == colcum[c-1:]
            colmask = np.logical_and(colmask, self.field[:, :-c+1: Channels.Placed.value])
        mask[:self.h-r+1, :self.w-c+1, 0] = np.logical_and(rowmask[:, :self.w-c+1], colmask[:self.h-r+1, :])

        if r != c:
            r, c = c, r
            if r == 1:
                rowmask = self.field[..., Channels.Placed.value]
            else:
                rowcum = self.field[..., Channels.Placed.value].cumsum(axis=0)
                rowmask = rowcum[:-r+1] == rowcum[r-1:]
                rowmask = np.logical_and(rowmask, self.field[:-r+1, :, Channels.Placed.value])
            if c == 1:
                colmask = self.field[..., Channels.Placed.value]
            else:
                colcum = self.field[..., Channels.Placed.value].cumsum(axis=1)
                colmask = colcum[:, :-c+1] == colcum[c-1:]
                colmask = np.logical_and(colmask, self.field[:, :-c+1: Channels.Placed.value])
            mask[:self.h-r+1, :self.w-c+1, 1] = np.logical_and(rowmask[:, :self.w-c+1], colmask[:self.h-r+1, :])
        else:
            mask[..., 1] = mask[..., 0]
        
        mask = np.concat([mask, mask], axis=-1)
        return torch.from_numpy(mask).contiguous().to(dtype=torch.bool).reshape(-1, 4)
    
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
        io_dir_mask = self.field[..., Channels.MachineDirection.value, np.newaxis] == directions
        io_mask = np.logical_and(io_mask, io_dir_mask)
        io_mask = torch.from_numpy(io_mask).to(dtype=torch.bool)
        
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
        cross_mask = np.expand_dims(cross_mask, axis=-1)
        cross_right_mask = np.logical_and(
            cross_mask,
            self.field[..., Channels.BeltRight.value] == 0
        )
        cross_up_mask = np.logical_and(
            cross_mask,
            self.field[..., Channels.BeltUp.value] == 0
        )
        cross_mask = np.concat([cross_up_mask, cross_right_mask, cross_up_mask, cross_right_mask], axis=-1)
        cross_mask = torch.from_numpy(cross_mask).to(dtype=torch.bool)

        place_mask = self.field[..., Channels.Placed.value] == 0
        place_mask = torch.from_numpy(place_mask).to(torch.bool).unsqueeze(-1).expand_as(io_mask)
        
        return (io_mask | cross_mask | place_mask).reshape(-1, 4)

    def get_spatial_power_mask(self) -> torch.Tensor:
        r, c = 2, 2
        mask = np.zeros((self.h, self.w), dtype=np.bool)
        if r == 1:
            rowmask = self.field[..., Channels.Placed.value]
        else:
            rowcum = self.field[..., Channels.Placed.value].cumsum(axis=0)
            rowmask = rowcum[:-r+1] == rowcum[r-1:]
            rowmask = np.logical_and(rowmask, self.field[:-r+1, :, Channels.Placed.value])
        if c == 1:
            colmask = self.field[..., Channels.Placed.value]
        else:
            colcum = self.field[..., Channels.Placed.value].cumsum(axis=1)
            colmask = colcum[:, :-c+1] == colcum[c-1:]
            colmask = np.logical_and(colmask, self.field[:, :-c+1: Channels.Placed.value])
        mask[:self.h-r+1, :self.w-c+1] = np.logical_and(rowmask[:, :self.w-c+1], colmask[:self.h-r+1, :])
        return torch.from_numpy(mask).contiguous().to(dtype=torch.bool).flatten()

    def get_states(self) -> Dict[str, Any]:
        spatial_states = torch.from_numpy(self.field[..., Channels.Placed.value:]).contiguous().to(dtype=torch.int8)
        spatial_types = torch.from_numpy(self.field[..., Channels.MachineType.value]).contiguous()
        return {
            'spatial_info': {
                'states': spatial_states,
                'types': spatial_types,
            },
            'spatial_hw': torch.tensor((self.h, self.w), dtype=torch.int32),
            'graph_nodes': torch.from_numpy(self.nodes),
            'edge_index': torch.from_numpy(self.edges),
            'node_lengths': torch.tensor(self.nodes.shape[0]),
        }

    def put_machine(self, machine_id, position, direction):
        machine_type = self.nodes[machine_id, NodeAttrs.MachineType.value]
        name, r, c, inputs, outputs = Machines[machine_type]
        if direction in [Directions.Right.value, Directions.Left.value]:
            r, c = c, r
        x, y = position
        if x + r > self.h or y + c > self.w:
            raise RuntimeError()
        for i in range(x, x + r):
            for j in range(y, y + c):
                if self.field[i, j, Channels.Placed.value] != 0:
                    raise RuntimeError()
        for i in range(x, x + r):
            for j in range(y, y + c):
                self.field[i, j, Channels.Placed.value] = 1
                self.field[i, j, Channels.IsMachine.value] = 1
                self.field[i, j, Channels.MachineType.value] = machine_type
                self.field[i, j, Channels.MachineDirection.value] = direction
                self.field[i, j, Channels.MachineId.value] = self.global_mid

        # rotate clockwise 90 first
        if direction in [Directions.Right.value, Directions.Left.value]:
            inputs = ((yy, c - xx) for xx, yy in inputs)
            outputs = ((yy, c - xx) for xx, yy in outputs)
        # rotate 180 then
        if direction in [Directions.Down.value, Directions.Left.value]:
            inputs = ((r - xx, c - yy) for xx, yy in inputs)
            outputs = ((r - xx, c - yy) for xx, yy  in inputs)
        for xx, yy in inputs:
            self.field[x + xx, y + yy, Channels.InputPort.value] = 1
        for xx, yy in outputs:
            self.field[x + xx, y + yy, Channels.OutputPort.value] = 1

        self.nodes[machine_id, NodeAttrs.Placed.value] = 1
        self.global_mid += 1
    
    def put_belt(self, position, direction):
        x, y = position
        if self.field[x, y, Channels.Placed.value] == 0:
            self.field[x, y, Channels.Placed.value] = 1
            self.field[x, y, Channels.IsBelt.value] = 1
            if direction in [Directions.Right.value, Directions.Left.value]:
                self.field[x, y, Channels.BeltRight.value] = 1 if direction == Directions.Right.value else -1
            elif direction in [Directions.Up.value, Directions.Down.value]:
                self.field[x, y, Channels.BeltUp.value] = 1 if direction == Directions.Up.value else -1

            hasStraight = False
            hasTurn = False
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
                if pointsTo:
                    if dd == direction:
                        hasStraight = True
                    else:
                        hasTurn = True
            if not hasStraight and hasTurn:
                self.field[x, y, Channels.BeltTurn.value] = 1

        elif self.field[x, y, Channels.InputPort.value] == 1 or self.field[x, y, Channels.OutputPort.value] == 1:
            if self.field[x, y, Channels.IsBelt.value]:
                raise RuntimeError()
            self.field[x, y, Channels.IsBelt.value] = 1
            if direction !=  self.field[x, y, Channels.MachineDirection.value]:
                raise RuntimeError()
            if direction in [Directions.Right.value, Directions.Left.value]:
                self.field[x, y, Channels.BeltRight.value] = 1 if direction == Directions.Right.value else -1
            elif direction in [Directions.Up.value, Directions.Down.value]:
                self.field[x, y, Channels.BeltUp.value] = 1 if direction == Directions.Up.value else -1
        
        elif self.field[x, y, Channels.IsBelt.value] == 1:
            if direction in [Directions.Right.value, Directions.Left.value]:
                if self.field[x, y, Channels.BeltRight.value] != 0:
                    raise RuntimeError()
                self.field[x, y, Channels.BeltUp.value] = 1 if direction == Directions.Right.value else -1
                self.field[x, y, Channels.BeltCross.value] = 1
            elif direction in [Directions.Up.value, Directions.Down.value]:
                self.field[x, y, Channels.BeltUp.value] = 1 if direction == Directions.Up.value else -1
                self.field[x, y, Channels.BeltCross.value] = 1
        
        else:
            raise RuntimeError()

    def put_power(self, position):
        x, y = position
        for i in (x, x+1):
            if i >= self.h:
                raise RuntimeError()
            for j in (y, y+1):
                if j >= self.w:
                    raise RuntimeError()
                if self.field[i, j, Channels.Placed.value]:
                    raise RuntimeError()
        
        for i in (x, x+1):
            for j in (y, y+1):
                self.field[i, j, Channels.Placed.value] = 1
                self.field[i, j, Channels.IsPower.value] = 1

        for i in range(x-5, x+7):
            if i < 0 or i >= self.h:
                continue
            for j in range(y-5, y+7):
                if j < 0 or j >= self.w:
                    continue
                self.field[i, j, Channels.Powered.value] = 1
                if self.field[i, j, Channels.IsMachine.value] == 1:
                    machine_id = self.field[i, j, Channels.MachineId.value]
                    self.nodes[machine_id, NodeAttrs.Powered.value] = 1
