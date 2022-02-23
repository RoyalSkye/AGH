import math
import torch
import os, random
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    start_time = time.time()
    model, _ = load_model(opts.model)
    assert model.is_agh, "For other problem, please refer to https://github.com/wouterkool/attention-learn-to-route/blob/master/eval.py"
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    print(results.tolist())
    print("Using {} strategy: Average cost: {} +- {}".format(opts.decode_strategy, torch.mean(results), torch.std(results) / math.sqrt(len(results))))
    print(">> End of validation within {:.2f}s".format(time.time()-start_time))


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    cost = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch_cost = []
        batch = move_to(batch, device)
        with torch.no_grad():
            # preprocess data
            bat_tw_left = batch['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(device)
            bat_tw_right = batch['departure']
            for f in model.fleet_info['order']:
                next_duration = torch.tensor(
                    model.fleet_info['next_duration'][model.fleet_info['precedence'][f]],
                    device=batch['type'].device).repeat(batch['loc'].size(0), 1)
                tw_right = bat_tw_right - torch.gather(next_duration, 1, batch['type'])
                tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)
                tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
                tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)
                duration = torch.tensor(model.fleet_info['duration'][f],
                                        device=batch['type'].device).repeat(batch['loc'].size(0), 1)
                fleet_bat = {'loc': batch['loc'], 'demand': batch['demand'][:, f - 1, :],
                             'distance': model.distance.expand(batch['loc'].size(0), len(model.distance)),
                             'duration': torch.gather(duration, 1, batch['type']),
                             'tw_right': tw_right, 'tw_left': tw_left,
                             'fleet': torch.full((batch['loc'].size(0), 1), f - 1)}
                if model.rnn_time:
                    model.pre_tw = None

                if opts.decode_strategy in ('sample', 'greedy'):
                    if opts.decode_strategy == 'greedy':
                        assert width == 0, "Do not set width when using greedy"
                        assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                            "eval_batch_size should be smaller than calc batch size"
                        batch_rep = 1
                        iter_rep = 1
                    elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                        assert opts.eval_batch_size == 1
                        assert width % opts.max_calc_batch_size == 0
                        batch_rep = opts.max_calc_batch_size
                        iter_rep = width // opts.max_calc_batch_size
                    else:
                        # sample for agh: for each fleet, sample batch_rep * iter_rep solutions,
                        # choose the best one, and go for next loop (fleet), otherwise exponential complexity!
                        batch_rep = width
                        iter_rep = 1
                    assert batch_rep > 0
                    sequences, costs, serve_time = model.sample_many(move_to(fleet_bat, device), batch_rep=batch_rep, iter_rep=iter_rep)
                    batch_size = len(costs)
                    batch_cost.append(costs.data.cpu().view(-1, 1))
                else:
                    assert opts.decode_strategy == 'bs'
                    assert opts.problem != 'agh', 'not supported currently!'

                    cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                        batch, beam_size=width,
                        compress_mask=opts.compress_mask,
                        max_calc_batch_size=opts.max_calc_batch_size
                    )

                # update tw_left
                bat_tw_left[model.fleet_info['precedence'][f] + 1] = torch.max(bat_tw_left[model.fleet_info['precedence'][f] + 1], serve_time[:, 1:])

            batch_cost = torch.cat(batch_cost, 1)  # [batch_size, 10]
            cost.append(batch_cost)

    return torch.cat(cost, 0).sum(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', default=["./data/agh/agh20_validation_seed4321.pkl", ], help="Filename of the dataset(s) to evaluate")
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--val_size', type=int, default=1000, help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0, help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=10, help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str, default='sample', choices=['sample', 'greedy', 'bs'],
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1, help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str, default="./data/epoch-50.pt")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=1000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true', help='Use multiprocessing to parallelize over multiple GPUs')

    # sample: eval_batch_size=1, width=1000, decode_strategy=sample
    opts = parser.parse_args()

    # Set the random seed
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    widths = opts.width if opts.width is not None else [0]

    for width in widths:
        for dataset_path in opts.datasets:
            eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
