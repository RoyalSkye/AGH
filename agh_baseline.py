# Clarke and Wright Savings
import copy
import math
import torch
import pickle
import argparse
import pprint as pp
from tqdm import tqdm
from utils import move_to, load_problem
from torch.utils.data import DataLoader


def inner(input_, problem, opt):
    NODE_SIZE, SPEED = 92, 110.0

    sequences = []
    state = problem.make_state(input_)
    ids = torch.arange(opt.val_size)[:, None]

    # calculate savings
    from_depot, to_depot = input_['loc'], NODE_SIZE * input_['loc']
    from_depot, to_depot = input_['distance'].gather(1, from_depot), input_['distance'].gather(1, to_depot)
    from_depot = from_depot[:, None, :].repeat(1, opt.graph_size, 1)
    to_depot = to_depot[:, :, None].repeat(1, 1, opt.graph_size)

    i_index = input_['loc'][:, :, None].repeat(1, 1, opt.graph_size)
    j_index = input_['loc'][:, None, :].repeat(1, opt.graph_size, 1)
    i_j = NODE_SIZE * i_index + j_index
    temp_distance = input_['distance'][:, None, :].expand(opt.val_size, opt.graph_size, NODE_SIZE * NODE_SIZE)
    i_j_distance = temp_distance.gather(2, i_j)

    savings_distance = from_depot + to_depot - i_j_distance

    tw_left, tw_right = input_['tw_left'][:, 1:], input_['tw_right'][:, 1:]
    savings_time = tw_left[:, None, :].repeat(1, opt.graph_size, 1) - tw_left[:, :, None].repeat(1, 1, opt.graph_size)
    savings_time = savings_time - input_['duration'][:, :, None].repeat(1, 1, opt.graph_size) - i_j_distance / SPEED

    savings = savings_distance - 0.03 * 60 * savings_time

    # select first nodes
    _, selected = tw_right.sort(dim=1)
    selected = 1 + selected[:, 0]

    state = state.update(selected)

    sequences.append(selected)

    # schedule following nodes
    i = 0
    while not (state.all_finished()):
        mask = state.get_mask()
        prev = state.prev_a - 1

        score = savings[ids, prev, :][:, 0, :]
        depot_score, _ = score.min(dim=1)
        depot_score = depot_score[:, None] - 1
        score = torch.cat((depot_score, score), dim=1)
        score[mask[:, 0, :]] = -math.inf

        _, selected = score.sort(descending=True)
        selected = selected[:, 0]

        state = state.update(selected)
        sequences.append(selected)
        i += 1

    cost, _ = problem.get_costs(input_, torch.stack(sequences, 1))

    return cost, state.serve_time


def CWS(dataset, opt, fleet_info, distance, problem):
    cost = []
    for bat in tqdm(DataLoader(dataset, batch_size=opt.val_size), disable=opt.no_progress_bar):
        bat_cost = []
        bat_tw_left = bat['arrival'].repeat(len(fleet_info['next_duration']) + 1, 1, 1).to(opts.device)
        bat_tw_right = bat['departure']
        for f in fleet_info['order']:
            # merge more data
            next_duration = torch.tensor(fleet_info['next_duration'][fleet_info['precedence'][f]],
                                         device=bat['type'].device).repeat(bat['loc'].size(0), 1)
            tw_right = bat_tw_right - torch.gather(next_duration, 1, bat['type'])
            tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)
            tw_left = bat_tw_left[fleet_info['precedence'][f]]
            tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)
            duration = torch.tensor(fleet_info['duration'][f],
                                    device=bat['type'].device).repeat(bat['loc'].size(0), 1)
            fleet_bat = {'loc': bat['loc'], 'demand': bat['demand'][:, f - 1, :],
                         'distance': distance.expand(bat['loc'].size(0), len(distance)),
                         'duration': torch.gather(duration, 1, bat['type']),
                         'tw_right': tw_right, 'tw_left': tw_left,
                         'fleet': torch.full((bat['loc'].size(0), 1), f - 1)}
            fleet_cost, serve_time = inner(move_to(fleet_bat, opt.device), problem, opt)
            bat_cost.append(fleet_cost.data.cpu().view(-1, 1))

            # update tw_left
            bat_tw_left[fleet_info['precedence'][f] + 1] = torch.max(
                bat_tw_left[fleet_info['precedence'][f] + 1], serve_time[:, 1:])

        bat_cost = torch.cat(bat_cost, 1)
        cost.append(bat_cost)  # (data_size, 10)

    cost = torch.cat(cost, 0)
    cost = cost.sum(1)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to load (ignores datadir)")
    parser.add_argument("--problem", type=str, default='agh',
                        help="only support airport ground handling in this code")
    parser.add_argument('--graph_size', type=int, default=20,
                        help="Sizes of problem instances (20, 50, 100)")
    parser.add_argument('--val_size', type=int, default=1000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    opts = parser.parse_args()

    opts.use_cuda = torch.cuda.is_available()
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    pp.pprint(vars(opts))

    # Figure out what's the problem
    problem_ = load_problem(opts.problem)

    val_dataset = problem_.make_dataset(size=opts.graph_size, num_samples=opts.val_size, filename=opts.filename)

    with open('problems/agh/fleet_info.pkl', 'rb') as file_:
        fleet_info_ = pickle.load(file_)
    with open('problems/agh/distance.pkl', 'rb') as file_:
        distance_ = pickle.load(file_)
    distance_ = torch.tensor(list(distance_.values()))

    CWS(val_dataset, opts, fleet_info_, distance_, problem_)
