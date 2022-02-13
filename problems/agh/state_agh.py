import torch
import pickle
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateAGH(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor
    tw_left: torch.Tensor
    tw_right: torch.Tensor
    distance: torch.Tensor
    duration: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_free_time: torch.Tensor
    serve_time: torch.Tensor
    tour: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    VEHICLE_CAPACITY = 1.0  # Hardcoded
    SPEED = 110.0
    NODE_SIZE = 92

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            cur_free_time=self.cur_free_time[key],
            tour=self.tour[key],
            serve_time=self.serve_time[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        loc = input['loc']
        demand = input['demand']
        tw_left, tw_right = input['tw_left'], input['tw_right']
        batch_distance = input['distance']

        batch_size, n_loc = loc.size()
        return StateAGH(
            coords=torch.cat((torch.zeros_like(loc[:, :1], device=loc.device), loc), dim=1),  # (batch, agh+1)
            demand=demand,  # (batch, agh)
            tw_left=tw_left,  # (batch, agh+1)
            tw_right=tw_right,  # (batch, agh+1)
            distance=batch_distance,  # (batch, NODE*NODE)
            duration=torch.cat((torch.zeros_like(loc[:, :1], device=loc.device), input['duration']), dim=1),  # (batch, agh+1)
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=torch.zeros_like(loc[:, :1], device=loc.device),  # Add step dimension
            cur_free_time=torch.full((batch_size, 1), -60, device=loc.device),  # max_distance=2501.98 / SPEED = 22.74m
            serve_time=torch.zeros(batch_size, n_loc+1, device=loc.device).float(),
            tour=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):
        # TODO: combine with beam search, complete in the future
        pass

    def update(self, selected):
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        distance_index = self.NODE_SIZE * self.coords[self.ids, self.prev_a] + cur_coord
        lengths = self.lengths + self.distance.gather(1, distance_index)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        # Increase capacity if depot is not visited, otherwise set to 0
        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]
        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        cur_free_time = (torch.max(
            self.cur_free_time + self.distance.gather(1, distance_index) / self.SPEED,
            self.tw_left[self.ids, selected])
                         + self.duration[self.ids, selected]) * (prev_a != 0).float() - 60 * (prev_a == 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        serve_time = self.serve_time.scatter_(1, selected, cur_free_time.float())

        tour = torch.cat((self.tour, selected), dim=1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1,
            cur_free_time=cur_free_time, tour=tour, serve_time=serve_time
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited,
        current free time and remaining capacity.
        !!! 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        """
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        pre_coord = self.coords[self.ids, self.prev_a]  # [batch_size, 1]
        all_coord = self.coords[:, 1:]
        distance_index = self.NODE_SIZE * pre_coord.expand(self.coords.size(0), self.coords.size(1)-1) + all_coord
        exceeds_tw = ((torch.max(
            self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1) +
            self.distance.gather(1, distance_index) / self.SPEED, self.tw_left[:, 1:])
            + self.duration[:, 1:]) > self.tw_right[:, 1:])[:, None, :]
        # Nodes that cannot be visited are already visited or too much demand to be served now or cannot satisfy tw
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap | exceeds_tw

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)  # [batch_size, 1, n_loc + 1]

    def construct_solutions(self, actions):
        # TODO: also combine with beam search, complete in the future
        pass
