#!/usr/bin/env python

import os
import re
import argparse
import time
import random
from models import *
from projectSolver import solveProject, extract_v_feature, extract_k_f_feature
from alns_cvrptw import by_vehicle_destroy_with_learning, construct_cplex_model
from datasets import *
from utils import create_datasets, save_sol, used_vehicle, show, seed_everything, pre_process, compute_incumbents_avg


def evaluate(args, plot=False):
    # Load model
    print(f">> strategy: {args.strategy}, destroy_degree: {args.destroy_degree}, timelimit: {args.timelimit}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = GCN(c_dim=4, x_dim=3, t_dim=2, k_f_dim=2, emb_dim=64, hidden=128, dropout=0.5).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=str(device))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(model)
    print("Loaded model from checkpoint, it has been trained {} epochs with {:.4f} loss and {:.4f} accuracy"
          .format(checkpoint['epoch'], checkpoint['best_loss'], checkpoint['best_acc']))

    files = os.listdir(args.data_folder)
    obj_list = []
    for _, file in enumerate(files):
        if os.path.splitext(file)[-1][1:] not in ["xlsx"]:
            print("Unsupported file detected!")
            continue
        path = os.path.join(args.data_folder, file)
        print(path)
        file_count = int(re.findall('\d+', path)[-1])
        loading_path = None
        # loading_path = os.path.join(args.data_folder, f"SHARED_DATA_{file_count}.json")

        # Run original 4 LNS with different heuristics.
        res = []
        seed_everything(args.seed)
        res = solveProject(args.prjCfgPath, path, args.opnFilePath, args.netFilePath, args.resultFilePath, learning=False, evaluate=False)

        # extract feature
        seed_everything(args.seed)
        file_map, shared_data, unshared_data = {}, {}, {}
        file_map[path] = 0  # file_map
        cvrptw, cvrptw_best = solveProject(args.prjCfgPath, path, args.opnFilePath, args.netFilePath, args.resultFilePath, learning=True, evaluate=True)  # id_map, veh_map
        cvrptw.model, cvrptw.lns_constraints = None, None
        t1 = time.time()
        mdl = construct_cplex_model(cvrptw, x_fixed=None, loading_path=loading_path)  # c, e
        print(f">> Construct model within {int(time.time() - t1)}s")
        extract_v_feature(cvrptw, cvrptw_best, sol=True)  # x, t
        extract_k_f_feature(cvrptw, sol=True)  # k_f

        print("LNS BEGIN")
        start_time, extra_time = time.time(), 0
        res_list = [cvrptw.obj]
        time_list = [0]
        pre_destroy = set()

        for iteration in range(args.iter):
            start_time1 = time.time()
            unshared_data[0] = {"x": cvrptw.data["x"], "t": cvrptw.data["t"], "k_f": cvrptw.data["k_f"],
                                "label": [[0] * args.iter], "accepted": [True] * args.iter}
            shared_data[0] = {"obj": cvrptw_best.obj, "id_map": cvrptw.id_map, "veh_map": cvrptw.veh_map,
                              "c": cvrptw.data["c"], "e": cvrptw.data["e"]}
            create_datasets(shared_data, unshared_data, file_map, output_folder=args.data_folder, file_count=file_count)
            dataset = CVPRTWDataset(iteration=1, dir_path=args.data_folder, file_count=file_count)
            data = dataset.__getitem__(0)
            if data is None:
                continue
            c, x, t, k_f, e_cv, e_vc, e_v_veh, label, obj = data
            c = c.to(device)
            x = x.to(device)
            t = t.to(device)
            k_f = k_f.to(device)
            e_cv = e_cv.to(device)
            e_vc = e_vc.to(device)
            e_v_veh = e_v_veh.to(device)
            label = label.unsqueeze(1).to(device)

            # make prediction
            x = model(c, x, t, k_f, e_cv, e_vc, e_v_veh)
            x = x.view(-1)
            print(x)

            if args.strategy == 1:
                # 1. only choose vehicle with score > 0
                x = torch.where(x > args.threshold, torch.FloatTensor([1]), torch.FloatTensor([0]))
                destroy_list = torch.nonzero(x, as_tuple=False).view(-1).tolist()
            elif args.strategy == 2:
                # 2. sample strategy for choosing vehicle
                num = torch.where(x > args.threshold)[0].size(0)
                x = torch.softmax(x, dim=0)
                destroy_list = set(torch.multinomial(x, num, replacement=True).tolist())
            elif args.strategy == 3:
                # 3. destroy_degree provided
                num = int(x.size(0) * args.destroy_degree)
                x = torch.softmax(x, dim=0)
                destroy_list = set(torch.multinomial(x, num, replacement=False).tolist())
            elif args.strategy == 4:
                # 4. Draws binary random numbers (0 or 1) from a Bernoulli distribution
                x = torch.sigmoid(x)
                bernoulli_res = torch.bernoulli(x)
                destroy_list = torch.nonzero(bernoulli_res, as_tuple=False).view(-1).tolist()
                # double-check
                destroy_degree = random.uniform(0.2, 0.7)
                num = int(x.size(0) * destroy_degree)
                if len(destroy_list) > num:
                    print(">> Adasample Modifying...")
                    destroy_list = random.sample(destroy_list, num)
            elif args.strategy == 5:
                # 5. change to disjoint decomposition
                num = int(x.size(0) * args.destroy_degree)
                x = torch.softmax(x, dim=0)
                disjoint_set = set(range(len(cvrptw.veh_map))) - pre_destroy
                if len(disjoint_set) < num:
                    destroy_list = disjoint_set
                    for i in disjoint_set:
                        x[i] = 0
                    ll = set(torch.multinomial(x, num - len(disjoint_set), replacement=False).tolist())
                    for i in ll:
                        destroy_list.add(i)
                    # pre_destroy = ll
                    pre_destroy = set(destroy_list)
                else:
                    for i in pre_destroy:
                        x[i] = 0
                    destroy_list = set(torch.multinomial(x, num, replacement=False).tolist())
                    # pre_destroy = destroy_list
                    pre_destroy = pre_destroy.union(destroy_list)

            print(destroy_list)
            inverse_veh_map = {value: key for key, value in cvrptw.veh_map.items()}
            destroy_id = [inverse_veh_map[key] for key in destroy_list]

            extra_time += time.time() - start_time1

            # clear old feature
            cvrptw.data["x"], cvrptw.data["t"], cvrptw.data["k_f"] = [], [], []

            # run LNS based on model prediction
            print("Model decides to destroy {}/{}={:.2f}% vehicle".format(len(destroy_id), len(cvrptw.vehicle_fleet),
                                                                          len(destroy_id) / len(
                                                                              cvrptw.vehicle_fleet) * 100))
            solution = by_vehicle_destroy_with_learning(cvrptw, destroy_id, mipgap=args.mipgap,
                                                        timelimit=args.timelimit)

            time_list.append(int(time.time() - start_time - extra_time))
            if solution is None:
                print("{} - Cplex doesn't find a solution within {}s".format(iteration + 1, args.timelimit))
                res_list.append(None)
                print("Time {}".format(time_list[-1]))
                extract_v_feature(cvrptw, cvrptw_best, sol=False)
                extract_k_f_feature(cvrptw, sol=False)
                continue
            new_sol_obj = int(solution.objective_value)
            res_list.append(new_sol_obj)
            print("Old -> New Solution objective: {} -> {}".format(cvrptw.obj, new_sol_obj))
            # new solution will always be accepted to avoid stagnation
            visit = {k: int(cvrptw.x[k].solution_value) for k in cvrptw.visit.keys()}
            t1 = {k: cvrptw.t[k].solution_value for k in cvrptw.time.keys()}
            cvrptw.set_solution(visit, t1)
            if not cvrptw.is_valid():
                save_sol(visit, t1, filepath="./error_sol.json")
                print("New solution doesn't pass constraint test!")
                exit(0)
            print("{} - Get New Solution, obj {}, used {} vehicles".format(iteration + 1, cvrptw.obj,
                                                                           used_vehicle(cvrptw)))
            # substitute current best solution
            if new_sol_obj < cvrptw_best.obj:
                cvrptw_best.set_solution(visit, t1)
                # save_sol(visit, t1)

            # update v feature
            extract_v_feature(cvrptw, cvrptw_best, sol=True)
            extract_k_f_feature(cvrptw, sol=True)
            print("Time {}".format(time_list[-1]))

        print("AVG Construct Time {}s".format(cvrptw.construct_time / args.iter))
        print("LNS END after {}s".format(int(time.time() - start_time)))
        print("BEST_OBJ = {}".format(cvrptw_best.obj))
        print("BEST_SOL used {} Vehicles".format(used_vehicle(cvrptw_best)))
        save_sol(cvrptw_best.visit, cvrptw_best.time, filepath="./BEST_sol.json")
        print(res_list)
        print(time_list)
        for obj1, obj2 in enumerate(res_list):
            print("{}: {}".format(obj1, obj2), end=" -> ")
        print(" ")

        res.append([cvrptw_best.obj, res_list, time_list])
        obj_list.append(res)

        for i, j in enumerate(res):
            print(args.label[i], j[0])
            print(j[1])
            print(j[2])

        if plot:
            # plot result for comparison
            filename = os.path.splitext(file)[0] + ".png"
            path = os.path.join(args.data_folder, filename)
            show([i[2] for i in res], [i[1] for i in res], args.label, args.title, "Time", "Obj", path)
            # new_fig
            filename = os.path.splitext(file)[0] + "_1.png"
            path = os.path.join(args.data_folder, filename)
            x = list(range(args.iter + 1))
            X, Y = pre_process([x, x], [i[1] for i in res])
            show(X, Y, args.label, args.title, "Iteration", "Obj", path)

    # AVG_fig: obj_list = [ [[3], [3]], ...]
    vehicle_random, forward_training = [obj_list[i][0][1] for i in range(len(obj_list))], [obj_list[i][-1][1] for i in range(len(obj_list))]
    avg_res = compute_incumbents_avg(vehicle_random, forward_training)
    # avg_res = [[] for i in range(len(obj_list[0]))]
    avg_time = [[] for i in range(len(obj_list[0]))]
    path = os.path.join(args.data_folder, 'AVG.png')
    for idx in range(args.iter + 1):
        # sum = [0 for i in range(len(obj_list[0]))]
        sum1 = [0 for i in range(len(obj_list[0]))]
        for i in obj_list:
            for j in range(len(i)):
                # sum[j] += i[j][1][idx]
                sum1[j] += i[j][2][idx]
        # for i, s in enumerate(sum):
        #     avg_res[i].append(s / len(obj_list))
        for i, s in enumerate(sum1):
            avg_time[i].append(s / len(obj_list))
    print(avg_time)

    if plot:
        x = list(range(args.iter + 1))
        X, Y = pre_process([x, x], avg_res)
        show(X, Y, args.label, args.title, "Iteration", "Obj", path)

    return [avg_res[-1][0], avg_res[-1][-1], avg_res[-1][0] - avg_res[-1][-1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Imitation Learning for CVRPTW with LNS')
    # Data parameters
    parser.add_argument('--data_folder', default="./data/Test1", help='folder with data files')
    parser.add_argument('--prjCfgPath', default="./test_200304/testPrj.cfg", help='folder with data files')
    parser.add_argument('--opnFilePath', default="./test_200304/operation_W.xlsx", help='folder with data files')
    parser.add_argument('--netFilePath', default="./test_200304/airport.net.xml", help='folder with data files')
    parser.add_argument('--resultFilePath', default="./test_200304/result.xlsx", help='folder with data files')
    parser.add_argument('--checkpoint', default="./models/checkpoint_FT_6_0.4.pth.tar", help='path to checkpoint, None if none.')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed.')
    parser.add_argument('--threshold', type=float, default=0., help='which var with activation before sigmoid will be chosen.')
    parser.add_argument('--iter', type=int, default=10, help='LNS run __ iterations for each instance.')
    parser.add_argument('--mipgap', type=float, default=0.1, help='hyper-parameter for CPLEX.')
    parser.add_argument('--timelimit', type=int, default=60, help='hyper-parameter for CPLEX.')
    parser.add_argument('--accept_rate', type=float, default=1.01, help='hyper-parameter for LNS.')
    parser.add_argument('--destroy_degree', type=float, default=0.4, help='hyper-parameter for LNS.')
    parser.add_argument('--strategy', type=int, default=3, help='which strategy to use during testing.')
    parser.add_argument('--title', default="20-Flights Vehicle LNS", help='for visualization.')
    parser.add_argument('--label', nargs='+', default=["Vehicle_Random LNS", "Forward_Training LNS"], help="for visualization.")

    args = parser.parse_args()

    evaluate(args, plot=True)
