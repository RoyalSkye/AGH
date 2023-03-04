#!/usr/bin/env python

import time, random
import torch.utils.data
from models import *
from datasets import *
import argparse
from utils import AverageMeter, save_checkpoint, clip_gradient


def train(args, training_set, model, optimizer, criterion):
    # one epoch training
    model.train()
    losses = AverageMeter()
    acc = AverageMeter()
    batch_time = AverageMeter()
    loss, correct, all, counter = 0, 0, 0, 0
    start_time = time.time()

    for _, idx in enumerate(training_set, start=1):
        data = dataset.__getitem__(idx)
        if data is None:
            continue
        counter += 1
        c, x, t, k_f, e_cv, e_vc, e_v_veh, label, obj = data

        # Convert to GPU, if available
        c = c.to(device)
        x = x.to(device)
        t = t.to(device)
        k_f = k_f.to(device)
        e_cv = e_cv.to(device)
        e_vc = e_vc.to(device)
        e_v_veh = e_v_veh.to(device)
        label = label.unsqueeze(1).to(device)  # for BCEWithLogitsLoss
        # label = label.long().to(device)  # for CrossEntropyLoss

        # print("label.size(): {}".format(label.size()))
        x = model(c, x, t, k_f, e_cv, e_vc, e_v_veh)
        loss += criterion(x, label)
        # L1 loss between 0 and score/logits
        # loss += torch.exp(-torch.sum(torch.abs(x)))
        # Encourage sum to 0
        # loss += torch.abs(torch.sum(x))

        # Back prop.
        if counter % args.batch_size == 0:
            optimizer.zero_grad()  # Set gradients to zero
            loss = loss/args.batch_size  # avg loss as the Loss returned by pytorch API
            loss.backward()  # From the loss we compute the new gradients
            if args.grad_clip is not None:
                clip_gradient(optimizer, args.grad_clip)
            # for name, parms in model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)

            optimizer.step()  # Update the parameters/weights
            losses.update(loss.detach().item())
            acc.update(correct/all)
            batch_time.update(time.time() - start_time)
            loss, correct, all, start_time = 0, 0, 0, time.time()

        pred = torch.where(x.squeeze(1) > args.threshold, torch.FloatTensor([1]), torch.FloatTensor([0]))
        correct += torch.eq(pred, label.squeeze(1)).sum().item()
        all += label.size(0)

        if counter % (args.batch_size * args.print_freq) == 0:
            print(torch.exp(-torch.sum(torch.abs(x))))
            print(x.squeeze(1))
            print("Epoch: {}/{} step: {}/{} Loss: {:.4f} Accuracy: {:.4f} Batch_time: {:.2f}s".format(
                    epoch, args.epochs, counter//args.batch_size, len(training_set)//args.batch_size,
                    losses.val, acc.val, batch_time.val))


def validate(args, validation_set, model, criterion):
    model.eval()
    losses = AverageMeter()
    acc = AverageMeter()
    correct, all = 0, 0
    start_time = time.time()
    with torch.no_grad():
        for _, idx in enumerate(validation_set, start=1):
            data = dataset.__getitem__(idx)
            if data is None:
                continue
            c, x, t, k_f, e_cv, e_vc, e_v_veh, label, obj = data

            # Convert to GPU, if available
            c = c.to(device)
            x = x.to(device)
            t = t.to(device)
            k_f = k_f.to(device)
            e_cv = e_cv.to(device)
            e_vc = e_vc.to(device)
            e_v_veh = e_v_veh.to(device)
            label = label.unsqueeze(1).to(device)

            x = model(c, x, t, k_f, e_cv, e_vc, e_v_veh)
            loss = criterion(x, label)

            losses.update(loss.item())
            pred = torch.where(x.squeeze(1) > args.threshold, torch.FloatTensor([1]), torch.FloatTensor([0]))
            correct += torch.eq(pred, label.squeeze(1)).sum().item()
            all += label.size(0)
        acc.update(correct/all)

    print("[*] EVA LOSS: {:.4f} Accuracy: {:.4f} Time: {:.2f}s".format(losses.avg, acc.val, time.time()-start_time))

    return losses.avg, acc.val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imitation Learning for CVRPTW with LNS')
    # Data parameters
    parser.add_argument('--data_folder', default="./data/Train1", help='folder with data files')
    parser.add_argument('--data_name', default="CVRPTW", help='base name shared by data files.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--grad_clip', type=float, default=2., help='clip gradients at an absolute value of.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--batch_size', type=int, default=5, help='accumulate gradients for batch_size samples.')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train.')
    parser.add_argument('--emb_dim', type=int, default=64, help='dimension of constraints and variables.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of activation of hidden layers.')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--print_freq', type=int, default=1, help='print training/validation stats every __ batches.')
    parser.add_argument('--threshold', type=float, default=0., help='which var with activation before sigmoid will be chosen.')
    parser.add_argument('--iter', type=int, default=30, help='LNS run __ iterations for each instance.')
    parser.add_argument('--update_criteria', type=int, default=5, help='adjust lr if epochs_since_improvement == update_criteria')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    print(device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = GCN(c_dim=4, x_dim=3, t_dim=2, k_f_dim=2, emb_dim=64, hidden=args.hidden_dim, dropout=args.dropout).to(device)
    # Adam is guaranteed to min the loss when the loss function is convex
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.checkpoint is None:
        start_epoch = 1
        best_loss, best_acc, epochs_since_improvement = 10 ** 6, 0, 0
    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss, best_acc = checkpoint['best_loss'], checkpoint['best_acc']

    print(model)
    print("model parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    # Do not support batch training now.
    t = time.time()
    lr = args.lr
    dataset = CVPRTWDataset(iteration=args.iter, dir_path=args.data_folder)
    print("[*] Load data successfully from {} within {:.2f}s".format(args.data_folder, time.time()-t))
    # train_loader = torch.utils.data.DataLoader(CVPRTWDataset(iteration=10, dir_path=args.data_folder), batch_size=1,
    #                                            shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(start_epoch, args.epochs + 1):
        # shuffle dataset every epoch
        idx = [i for i in range(len(dataset))]
        random.shuffle(idx)
        end = int(len(idx) * 0.7)
        training_set = idx[: end]
        validation_set = idx[end:]
        train(args, training_set, model, optimizer, criterion)
        l, a = validate(args, validation_set, model, criterion)

        # Record result and adjust lr
        # is_best = l < best_loss and a > best_acc
        is_best = l < best_loss
        best_loss = min(best_loss, l)
        best_acc = max(best_acc, a)
        if not is_best:
            epochs_since_improvement += 1
            print("[!] Epochs since last improvement: {}".format(epochs_since_improvement))
        else:
            epochs_since_improvement = 0
        if epochs_since_improvement > 0 and epochs_since_improvement % args.update_criteria == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr * 0.5)
            # adjust_learning_rate(optimizer, 0.5)

        # Save checkpoint
        save_checkpoint(args.data_name, epoch, model, optimizer, best_loss, best_acc, is_best, epochs_since_improvement)
