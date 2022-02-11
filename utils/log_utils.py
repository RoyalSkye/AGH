import os
import sys
import matplotlib.pyplot as plt


class Logger(object):
    """
    Save training process to log file with simple plot function.
    """
    def __init__(self, fpath=None, resume=False):
        self.file = None
        self.resume = resume
        if fpath is not None:
            if resume:
                print(f">> Resuming log {fpath}")
                self.file = open(fpath, 'a')
            else:
                print(f">> New log {fpath}")
                self.file = open(fpath, 'w')
        else:
            print(f">> Log to sys.stdout")
            self.file = sys.stdout

    def write(self, strr):
        self.file.write(strr)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('loss: {}, grad norm: {}, clipped: {}'.format(reinforce_loss, grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
