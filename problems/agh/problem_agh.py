import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from problems.agh.state_agh import StateAGH


class AGH(object):

    NAME = 'agh'  # Airport Ground Handling

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)
    SPEED = 110.0
    NODE_SIZE = 92

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
                       torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
                       sorted_pi[:, -graph_size:]
               ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -AGH.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)  # demand sorted as pi

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= AGH.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # (batch_size, agh_size + 1)
        loc = torch.cat((torch.zeros_like(dataset['loc'][:, :1]), dataset['loc']), dim=1)
        loc = loc.gather(1, pi)
        distance_index = AGH.NODE_SIZE * torch.cat((torch.zeros_like(loc[:, :1]), loc),
                                                   dim=1) + torch.cat((loc, torch.zeros_like(loc[:, :1])), dim=1)
        batch_distance = dataset['distance']

        # check the constraint of time window
        ids = torch.arange(batch_size, dtype=torch.int64, device=pi.device)[:, None]  # Add steps dimension
        time_distance = batch_distance / AGH.SPEED
        time_distance = time_distance.gather(1, distance_index)  # (batch_size, len(pi) + 1)
        # if the vehicle go back to the depot, time should be reset to 0, means this is a new vehicle
        cur_time = torch.full_like(pi[:, 0:1], -60)
        duration = torch.cat((torch.zeros_like(dataset['duration'][:, :1],
                                               device=dataset['duration'].device), dataset['duration']), dim=1)
        for i in range(pi.size(1)):
            cur_time = (torch.max(cur_time + time_distance[:, i:i+1], dataset['tw_left'][ids, pi[:, i:i+1]])
                        + duration[ids, pi[:, i:i+1]]) * (pi[:, i:i+1] != 0).float() - 60 * (pi[:, i:i+1] == 0).float()
            assert (cur_time <= dataset['tw_right'][ids, pi[:, i:i+1]] + 1e-5).all()

        return batch_distance.gather(1, distance_index).sum(1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return AGHDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateAGH.initialize(*args, **kwargs)

    @staticmethod
    def beam_search():
        # TODO: complete this function
        pass


def make_instance(args):
    loc, arrival, departure, type_, demand, *args = args
    return {
        'loc': torch.tensor(loc, dtype=torch.long),
        'arrival': torch.tensor(arrival, dtype=torch.float),
        'departure': torch.tensor(departure, dtype=torch.float),
        'type': torch.tensor(type_, dtype=torch.long),
        'demand': torch.tensor(demand, dtype=torch.float)
    }


class AGHDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, fleet_size=10):
        super(AGHDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
            n_gate, n_hour, n_min, prob = 91, 24, 60, np.load('problems/agh/arrival_prob.npy')
            loc = 1 + np.random.choice(n_gate, size=(num_samples, size))
            arrival = 60 * np.random.choice(n_hour, size=(num_samples, size),
                                            p=prob) + np.random.randint(0, n_min, size=(num_samples, size))
            stay = torch.tensor([30, 34, 33]).repeat(num_samples, 1)
            type_ = torch.tensor(np.random.randint(0, 3, size=(num_samples, size)), dtype=torch.long)
            departure = torch.tensor(arrival) + torch.gather(stay, 1, type_)
            demand = np.random.randint(1, 10, size=(num_samples, fleet_size, size)) / CAPACITIES[size]
            data = list(zip(loc.tolist(), arrival.tolist(), departure.tolist(), type_.tolist(), demand.tolist()))

        self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
