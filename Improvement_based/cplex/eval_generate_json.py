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


def evaluate(args, path, plot=False):
    # Load model
    # print(f">> strategy: {args.strategy}, destroy_degree: {args.destroy_degree}, timelimit: {args.timelimit}")    
    seed_everything(args.seed)
    file_count = int(re.findall('\d+', path)[-1])
    loading_path = None
    # loading_path = os.path.join(args.data_folder, f"SHARED_DATA_{file_count}.json")
    
    # extract feature
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
        break
        dataset = CVPRTWDataset(iteration=1, dir_path=args.data_folder, file_count=file_count)
        data = dataset.__getitem__(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Imitation Learning for CVRPTW with LNS')
    # Data parameters
    parser.add_argument('--data_folder', default="./data/200_Test2", help='folder with data files')
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

    from multiprocessing import Pool
    pool = Pool(processes = 10)
    files = os.listdir(args.data_folder)
    for _, file in enumerate(files):
        if os.path.splitext(file)[-1][1:] not in ["xlsx"]:
            print("Unsupported file detected!")
            continue
        path = os.path.join(args.data_folder, file)
        print(path)
        pool.apply_async(evaluate, args=(args, path))       
    pool.close()
    pool.join()
