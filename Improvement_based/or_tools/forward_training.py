import os
import copy
from train import *
from projectSolver import solveProject, extract_k_f_feature, extract_v_feature, lns
from utils import create_datasets, seed_everything, init_args
from alns_cvrptw import by_vehicle_destroy_with_learning, shuffle_vehicle_fleet, construct_cplex_model, construct_or_tools_model
from eval import evaluate


def single_batch_train(training_set, model, optimizer, criterion):
    # one epoch training
    model.train()
    losses = AverageMeter()
    correct, all = 0, 0

    for _, idx in enumerate(training_set, start=1):
        data = dataset.__getitem__(idx)
        if data is None:
            continue
        c, x, t, k_f, e_cv, e_vc, e_v_veh, label, obj = data
        label = label.unsqueeze(1)  # for BCEWithLogitsLoss
        # label = label.long()  # for CrossEntropyLoss

        optimizer.zero_grad()
        x = model(c, x, t, k_f, e_cv, e_vc, e_v_veh)
        loss = criterion(x, label)
        loss.backward()
        optimizer.step()

        losses.update(loss.detach().item())
        pred = torch.where(x.detach().squeeze(1) >= 0, torch.FloatTensor([1]), torch.FloatTensor([0]))
        correct += torch.eq(pred, label.squeeze(1)).sum().item()
        all += label.size(0)

    print(x.view(-1))

    return losses.avg, correct/all


def mini_batch_train(training_set, model, optimizer, criterion, batch_size=16):
    # one epoch training
    model.train()
    losses = AverageMeter()
    correct, all = 0, 0
    loss, num_batch = 0, 0

    for _, idx in enumerate(training_set, start=1):
        data = dataset.__getitem__(idx)
        if data is None:
            continue
        c, x, t, k_f, e_cv, e_vc, e_v_veh, label, obj = data
        label = label.unsqueeze(1)  # for BCEWithLogitsLoss
        # label = label.long()  # for CrossEntropyLoss

        x = model(c, x, t, k_f, e_cv, e_vc, e_v_veh)
        loss += criterion(x, label)
        num_batch += 1

        losses.update(loss.detach().item())
        pred = torch.where(x.detach().squeeze(1) >= 0, torch.FloatTensor([1]), torch.FloatTensor([0]))
        correct += torch.eq(pred, label.squeeze(1)).sum().item()
        all += label.size(0)

        if num_batch == batch_size:
            optimizer.zero_grad()
            loss = loss / batch_size
            loss.backward()
            optimizer.step()
            loss = loss.detach()
            loss, num_batch = 0, 0

    print(x.view(-1))

    return losses.avg, correct/all


if __name__ == "__main__":
    dir = "./data/20_Train/"
    val_dir = "./data/20_Val/"
    prjCfgPath = "./test_200304/testPrj.cfg"
    opnFilePath = "./test_200304/operation_W.xlsx"
    netFilePath = "./test_200304/airport.net.xml"
    resultFilePath = "./test_200304/result.xlsx"
    files = os.listdir(dir)
    val_files = os.listdir(val_dir)

    lr = 0.0001
    epochs = 10
    batch_size = 16
    hidden_dim = 128
    destroy_degree = 0.4
    mipgap = 0.1
    timelimit = 25
    iters = 5  # run {iters} iterations
    run_random = 10  # run {run_random} Vehicle Random LNS, choose the best
    seed = 2020

    seed_everything(seed)
    model = GCN(c_dim=4, x_dim=3, t_dim=2, k_f_dim=2, emb_dim=64, hidden=hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print(model)
    print("model parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        counter = 0
        data, id_map = {}, {}
        file_map, state, best_state = {}, {}, {}
        for file in files:
            if os.path.splitext(file)[-1][1:] not in ["xlsx"]:
                continue
            path = os.path.join(dir, file)
            file_map[path] = counter
            cvrptw, cvrptw_best = solveProject(prjCfgPath, path, opnFilePath, netFilePath, resultFilePath, learning=True, evaluate=True)
            cvrptw.vehicle_fleet = [(k, f) for f in cvrptw.args["fleets"] for k in cvrptw.args["vehicles"][f]]
            cvrptw.model, cvrptw.lns_constraints = None, None  # needed when run LNS for more than one iter, otherwise, c/e is None
            mdl = construct_or_tools_model(cvrptw, x_fixed=None)
            # mdl = construct_cplex_model(cvrptw, x_fixed=None)  # c, e
            extract_v_feature(cvrptw, cvrptw_best, sol=True)  # x, t
            extract_k_f_feature(cvrptw, sol=True)  # k_f
            state[counter], best_state[counter] = cvrptw, cvrptw_best
            counter += 1

        for iter in range(iters):
            shared_data, unshared_data = {}, {}
            files = os.listdir(dir)
            for file in files:
                if os.path.splitext(file)[-1][1:] not in ["xlsx"]:
                    continue
                path = os.path.join(dir, file)
                print(path)
                counter = file_map[path]
                cvrptw, cvrptw_best = state[counter], best_state[counter]
                best_obj, best_res = 10 ** 6, None
                for count in range(run_random):
                    shuffle_vehicle_fleet(cvrptw, shuffle="RANDOM")
                    destroy_num = int(len(cvrptw.vehicle_fleet) * destroy_degree)
                    destroy_id = cvrptw.vehicle_fleet[: destroy_num]
                    label = [0] * len(cvrptw.veh_map)
                    for i in destroy_id:
                        label[cvrptw.veh_map[i[0], i[1]]] = 1
                    solution = by_vehicle_destroy_with_learning(cvrptw, destroy_id, mipgap=mipgap, timelimit=timelimit)
                    if solution is None:
                        print(">> No solution.")
                        continue
                    tmp_cvrptw = copy.deepcopy(cvrptw)
                    # visit = {k: int(cvrptw.x[k].solution_value) for k in cvrptw.visit.keys()}
                    # t1 = {k: cvrptw.t[k].solution_value for k in cvrptw.time.keys()}
                    visit = {k: solution.Value(cvrptw.x[k]) for k in cvrptw.visit.keys()}
                    t1 = {k: solution.Value(cvrptw.t[k]) for k in cvrptw.time.keys()}
                    tmp_cvrptw.set_solution(visit, t1)
                    obj = tmp_cvrptw.objective()
                    if not tmp_cvrptw.is_valid():
                        print(">> Invalid solution.")
                        continue
                    cvrptw.data["accepted"] = [True] if obj < cvrptw.obj else [False]
                    res = [cvrptw.id_map, cvrptw.veh_map, copy.deepcopy(cvrptw.data), obj]
                    res[2]["label"] = [label]
                    print("Old -> New Solution objective: {} -> {}".format(cvrptw.obj, obj))
                    best_res = res if res[-1] < best_obj else best_res
                    best_obj = min(best_obj, res[-1])
                id_map, veh_map, data, obj = best_res

                print("\n>> After {} Vehicle Random LNS: {} -> {}".format(run_random, cvrptw.obj, obj))
                unshared_data[counter] = {"x": data["x"], "t": data["t"], "k_f": data["k_f"], "label": data["label"], "accepted": data["accepted"]}
                shared_data[counter] = {"obj": obj, "id_map": id_map, "veh_map": veh_map, "c": data["c"], "e": data["e"]}
            create_datasets(shared_data, unshared_data, file_map, output_folder=dir)

            # train a model
            start = time.time()
            dataset = CVPRTWDataset(iteration=1, dir_path=dir)
            print("[*] Load data successfully from {} within {:.2f}s".format(dir, time.time() - start))
            start = time.time()
            # shuffle dataset every epoch
            idx = [i for i in range(len(dataset))]
            random.shuffle(idx)
            # end = int(len(idx) * 0.7)
            # validation_set = idx[end:]
            training_set = idx[:]
            loss, acc = mini_batch_train(training_set, model, optimizer, criterion, batch_size=batch_size)
            print("Iter: {}/{} avg_loss: {:.4f} avg_acc: {:.2f} Batch_time: {:.2f}s".format(iter+1, iters, loss, acc * 100, time.time()-start))

            # make predictions and transfer to next state
            model.eval()
            dataset1 = CVPRTWDataset(iteration=1, dir_path=dir, return_all=True)
            for file in files:
                if os.path.splitext(file)[-1][1:] not in ["xlsx"]:
                    continue
                path = os.path.join(dir, file)
                counter = file_map[path]
                cvrptw, cvrptw_best = state[counter], best_state[counter]
                data = dataset1.__getitem__(counter)
                c, x, t, k_f, e_cv, e_vc, e_v_veh, _, _ = data

                x = model(c, x, t, k_f, e_cv, e_vc, e_v_veh)
                x = x.view(-1)
                num = int(x.size(0) * destroy_degree)
                x = torch.softmax(x, dim=0)
                destroy_list = set(torch.multinomial(x, num, replacement=False).tolist())
                print(destroy_list)

                inverse_veh_map = {value: key for key, value in cvrptw.veh_map.items()}
                destroy_id = [inverse_veh_map[key] for key in destroy_list]
                # clear old feature
                cvrptw.data["x"], cvrptw.data["t"], cvrptw.data["k_f"], cvrptw.data["label"], cvrptw.accepted = [], [], [], [], []
                # run LNS based on model prediction
                solution = by_vehicle_destroy_with_learning(cvrptw, destroy_id, mipgap=mipgap, timelimit=timelimit)
                if solution is None:
                    print(">> Cplex doesn't find a solution.")
                    extract_v_feature(cvrptw, cvrptw_best, sol=False)
                    extract_k_f_feature(cvrptw, sol=False)
                else:
                    # new_sol_obj = int(solution.objective_value)
                    new_sol_obj = int(solution.ObjectiveValue())
                    print("{}: Old -> New Solution objective: {} -> {}".format(file, cvrptw.obj, new_sol_obj))
                    # visit = {k: int(cvrptw.x[k].solution_value) for k in cvrptw.visit.keys()}
                    # t1 = {k: cvrptw.t[k].solution_value for k in cvrptw.time.keys()}
                    visit = {k: solution.Value(cvrptw.x[k]) for k in cvrptw.visit.keys()}
                    t1 = {k: solution.Value(cvrptw.t[k]) for k in cvrptw.time.keys()}
                    cvrptw.set_solution(visit, t1)
                    if not cvrptw.is_valid():
                        print(">> New solution doesn't pass constraint test!")
                        exit(0)
                    if new_sol_obj < cvrptw_best.obj:
                        cvrptw_best.set_solution(visit, t1)
                    # update v feature
                    extract_v_feature(cvrptw, cvrptw_best, sol=True)
                    extract_k_f_feature(cvrptw, sol=True)

                state[counter], best_state[counter] = cvrptw, cvrptw_best

        model_path = save_checkpoint("FT", epoch + 1, model, optimizer, 0, 0, False, 0)

        # validation
        args = init_args()
        args.data_folder = val_dir
        args.checkpoint = model_path
        args.iter = iters
        args.seed = seed
        args.mipgap = mipgap
        args.timelimit = timelimit
        args.destroy_degree = destroy_degree
        args.strategy = 3
        obj1, obj2, obj3 = evaluate(args, plot=False)
        print(f">> Validation for Epoch {epoch + 1}, run {iters} iters: {obj1} -> {obj2} = {obj3}")

