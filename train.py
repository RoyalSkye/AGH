import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    if model.is_agh:
        cost = cost.sum(1)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost


def rollout(model, dataset, opts, val_method="greedy"):
    # Put in greedy evaluation mode!
    set_decode_type(model, val_method)
    model.eval()

    if model.is_agh:
        cost = []
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
            bat_cost = []
            bat_tw_left = bat['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(opts.device)
            bat_tw_right = bat['departure']
            for f in model.fleet_info['order']:
                # merge more data
                next_duration = torch.tensor(
                    model.fleet_info['next_duration'][model.fleet_info['precedence'][f]],
                    device=bat['type'].device).repeat(bat['loc'].size(0), 1)
                tw_right = bat_tw_right - torch.gather(next_duration, 1, bat['type'])
                tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)
                tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
                tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)
                duration = torch.tensor(model.fleet_info['duration'][f],
                                        device=bat['type'].device).repeat(bat['loc'].size(0), 1)
                fleet_bat = {'loc': bat['loc'], 'demand': bat['demand'][:, f - 1, :],
                             'distance': model.distance.expand(bat['loc'].size(0), len(model.distance)),
                             'duration': torch.gather(duration, 1, bat['type']),
                             'tw_right': tw_right, 'tw_left': tw_left,
                             'fleet': torch.full((bat['loc'].size(0), 1), f - 1)}
                if model.rnn_time:
                    model.pre_tw = None

                # evaluate
                with torch.no_grad():
                    fleet_cost, _, serve_time = model(move_to(fleet_bat, opts.device))
                bat_cost.append(fleet_cost.data.cpu().view(-1, 1))

                # update tw_left
                bat_tw_left[model.fleet_info['precedence'][f] + 1] = torch.max(
                    bat_tw_left[model.fleet_info['precedence'][f] + 1], serve_time[:, 1:])
            bat_cost = torch.cat(bat_cost, 1)
            cost.append(bat_cost)  # [batch_size, 10]
        return torch.cat(cost, 0)  # [dataset, 10]

    def eval_model_bat(batch):
        with torch.no_grad():
            cost_, _ = model(move_to(batch, opts.device))
        return cost_.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        if model.is_agh:
            train_batch_agh(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
        else:
            train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict(),
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    # save_log
    with open(os.path.join(opts.save_dir, 'validate_log.txt'), 'a') as f:
        f.write('Validating Epoch {}, Validation avg_cost: {}\n'.format(epoch, avg_reward))
        f.write('\n')

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch_agh(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    assert bl_val is not None
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device)

    # train on batch data
    set_decode_type(model, "sampling")
    bat_tw_left = x['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1)  # [6, batch_size, graph_size]
    bat_tw_right = x['departure']  # [batch_size, graph_size]
    fleet_cost_together, log_likelihood_together, fleet_cost_list, log_likelihood_list = None, None, [], []

    for f in model.fleet_info['order']:
        # merge more data
        next_duration = torch.tensor(model.fleet_info['next_duration'][model.fleet_info['precedence'][f]],
                                     device=x['type'].device).repeat(x['loc'].size(0), 1)  # [batch_size, 3]
        tw_right = bat_tw_right - torch.gather(next_duration, 1, x['type'])  # [batch_size, graph_size]
        tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)  # [batch_size, graph_size+1]
        tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
        tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)  # [batch_size, graph_size+1]
        duration = torch.tensor(model.fleet_info['duration'][f], device=x['type'].device).repeat(x['loc'].size(0), 1)  # [batch_size, 3]
        fleet_bat = {'loc': x['loc'], 'demand': x['demand'][:, f - 1, :],
                     'distance': model.distance.expand(x['loc'].size(0), len(model.distance)),
                     'duration': torch.gather(duration, 1, x['type']),
                     'tw_right': tw_right, 'tw_left': tw_left,
                     'fleet': torch.full((x['loc'].size(0), 1), f - 1)}

        if model.rnn_time:
            model.pre_tw = None

        # solve
        # fleet_cost/log_likelihood: [batch_size]
        fleet_cost, log_likelihood, serve_time = model(move_to(fleet_bat, opts.device))

        # update model or information
        fleet_cost_list.append(fleet_cost)
        log_likelihood_list.append(log_likelihood)
        if fleet_cost_together is None:
            fleet_cost_together, log_likelihood_together = fleet_cost, log_likelihood
        else:
            fleet_cost_together = fleet_cost_together + fleet_cost
            log_likelihood_together = log_likelihood_together + log_likelihood

        # update tw_left
        bat_tw_left[model.fleet_info['precedence'][f] + 1, :, :] = torch.max(
            bat_tw_left[model.fleet_info['precedence'][f] + 1, :, :], serve_time[:, 1:])

        loss = ((fleet_cost_list[0] - bl_val[:, 0]) * log_likelihood_list[0]).mean()
        for i in range(1, len(fleet_cost_list)):
            loss += ((fleet_cost_list[i] - bl_val[:, i]) * log_likelihood_list[i]).mean()
        loss = loss / len(fleet_cost_list)

        optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        optimizer.step()

        # Logging
        if step % int(opts.log_step) == 0:
            log_values(fleet_cost_together, grad_norms, epoch, batch_id, step, log_likelihood_together, loss, 0, tb_logger, opts)


def train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
