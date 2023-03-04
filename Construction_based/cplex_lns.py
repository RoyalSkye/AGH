import os, sys, copy, time, json
import argparse, math, random
import torch
import pickle
import numpy as np
import pprint as pp
from tqdm import tqdm
from torch.utils.data import DataLoader
from docplex.mp.model import Model
from utils import move_to, load_problem


class CVRPTW(object):
    """
    Solution class for the CVRPTW problem. It has two data members, nodes, and edges.
    nodes is a list of node tuples: (id, coords). The edges data member, then, is
    a mapping from each node to their only outgoing node.
    """

    def __init__(self, args={}, init_solution=None):
        """
        ** vehicle capacity and speed are fixed to 1.0 and 110.0 **
        args = {"customers": customers, - [1,2,...,30]
            "fleets": fleets, - [1,2,...,10]
            "start_early": start_ea, - {1:19800,2:19620,...,50:0}
            "start_late": start_la, - {1:21600,2:21600,...,50:100000000}
            "nodes": nodes, - [1,2,...,50]
            "nodes_fleet": nodes_F, - {1:[1,2,...,30,31,32], 2: [1,2,...,30,33,34], 10: [1,2,...,30,49,50]}
            "travel_time": tau, duration(task on i) + travel time(from i to j) (i,j,f) - {(1,2,1):136.09636363636363,(1,2,2):...,(1,50,10):}
            "vehicles": vehicles, - {1:[1,2,...,20],2:[1,2,...,20],...,10:[1,2,...,20]}
            "distance": distance, - {(1,1):0,(1,2):282.96999999999997,...,(50,50):0}
            "demand_operation": D, - {1:[0,...(30),0],1:[],...,10:[]}
            "capacity_fleet": Q, - {1:[capacity, repeat*20],...,10[]}
            "duration": S_time, - {1:[0,...(30+20),0],2:[],...,10:[]}
            "precedence": precedence - {1:[[1, 2],[1, 3],[9, 10]],2:[],...,30:[]}
            "flight_type": flight_type - {1: 1, 2: 3, 3: 3, 4: 3, ..., 20: 2}
        }
        init_solution = {"x": x,
            "t": t
        }
        """
        super(CVRPTW).__init__()
        self.args = args
        self.obj = 10 ** 8
        self.model = None
        self.lns_constraints = None
        self.construct_time = 0
        self.vehicle_fleet = [(k, f) for f in self.args["fleets"] for k in self.args["vehicles"][f]]
        if init_solution is not None:
            self.set_solution(init_solution["x"], init_solution["t"])
        else:
            # decision variable
            self.visit = {(i, j, k, f): 0 for f in self.args["fleets"] for i in self.args["nodes_fleet"][f] for j in self.args["nodes_fleet"][f] for k in self.args["vehicles"][f] if i != j}
            self.time = {(i, f): 0.0 + args["start_early"][i] for f in self.args["fleets"] for i in self.args["nodes_fleet"][f]}  # the start time of operation f at pos i
            # recorded variable
            for f in self.args["fleets"]:
                self.time[len(self.args["customers"]) + f * 2 - 1, f] = 0
                self.time[len(self.args["customers"]) + f * 2, f] = 10 ** 8

    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        """
        The objective function is simply the sum of all individual edge lengths, using the rounded Euclidean norm.
        """
        n = len(self.args["customers"])
        obj = sum(self.args["distance"][i, j] * self.visit[i, j, k, f] for i, j, k, f in self.visit)
        return int(obj)

    def get_solution(self):
        return {"x": self.visit, "t": self.time}

    def set_solution(self, visit, t):
        self.visit = visit
        self.time = t
        self.obj = self.objective()

    def used_vehicle(self):
        vehicle_distance = {(k, f): 0 for f in self.args["fleets"] for k in self.args["vehicles"][f]}
        for key, value in self.visit.items():
            if value != 1:
                continue
            i, j, k, f = key[0], key[1], key[2], key[3]
            vehicle_distance[(k, f)] += self.args["distance"][(i, j)]
        count = {f: 0 for f in self.args["fleets"]}
        for key, value in vehicle_distance.items():
            if value != 0:
                count[key[-1]] += 1
        return count


def get_initial_sol(args):
    n, F, speed = len(args["customers"]), len(args["fleets"]), 110.0

    indices_tau = {(i, f) for f in args["fleets"] for i in args["nodes_fleet"][f]}
    # initial solution
    t = {(i, f): 0 + args["start_early"][i] for i, f in indices_tau}
    # 1 calculate start time of each fleet according to precedence
    for i in args["customers"]:
        for pair in args["precedence"][i]:
            t[i, pair[1]] = max(t[i, pair[1]], t[i, pair[0]]+args["duration"][pair[0]][i-1])
    for f in args["fleets"]:
        t[n + f * 2 - 1, f], t[n + 2 * f, f] = -60, 10 ** 8

    # calculate the vehicle number according to time and capacity, generate the route at the same time
    # 1 time window
    vehicle_number_approximation = {f: 0 for f in args["fleets"]}
    time_vehicle = {f: {i: 0 for i in args["customers"]} for f in args["fleets"]}
    route = dict()
    for f in args["fleets"]:
        vehicle_count = 1
        while 0 in list(time_vehicle[f].values()):
            route[f, vehicle_count] = []
            current_time = 0
            previous_flight = 0
            for j in range(n):
                if time_vehicle[f][j + 1] == 0:
                    current_time = t[j + 1, f] + args["duration"][f][j]
                    route[f, vehicle_count].append(j + 1)
                    time_vehicle[f][j + 1] = vehicle_count
                    previous_flight = j + 1
                    break
            nearest_flight = 1
            while nearest_flight != 0:
                nearest_distance = 10 ** 6
                nearest_flight = 0
                for j in range(n):
                    if time_vehicle[f][j + 1] == 0:
                        wait = t[j + 1, f] - current_time - args["distance"][previous_flight, j + 1] / speed
                        # if 0 < wait < 10 * 60:
                        if wait > 0 and nearest_flight == 0:
                            nearest_distance = args["distance"][previous_flight, j + 1] / speed
                            nearest_flight = j + 1
                        elif 0 < wait < 1 and nearest_distance > args["distance"][previous_flight, j + 1] / speed:
                            nearest_distance = args["distance"][previous_flight, j + 1] / speed
                            nearest_flight = j + 1
                if nearest_flight != 0:
                    current_time = t[nearest_flight, f] + args["duration"][f][nearest_flight - 1]
                    route[f, vehicle_count].append(nearest_flight)
                    time_vehicle[f][nearest_flight] = vehicle_count
                    previous_flight = nearest_flight

            vehicle_count += 1
        vehicle_number_approximation[f] = np.max(list(time_vehicle[f].values()))
    # 2 capacity
    for f in args["fleets"]:
        if args["capacity_fleet"][f] != 0:
            for k in range(1, vehicle_number_approximation[f] + 1):
                temp_demand = np.cumsum([args["demand_operation"][f][i-1] for i in route[f, k]])
                temp_route = route[f, k]
                temp_check = temp_demand < args["capacity_fleet"][f][k]
                false_ind = list(np.where(temp_check == False)[0])
                temp_k = k
                while False in temp_check:
                    route[f, temp_k] = temp_route[:min(false_ind)]
                    vehicle_number_approximation[f] += 1
                    route[f, vehicle_number_approximation[f]] = temp_route[min(false_ind):max(false_ind) + 1]
                    temp_k = vehicle_number_approximation[f]
                    temp_route = temp_route[min(false_ind):max(false_ind) + 1]
                    temp_demand = temp_demand - temp_demand[min(false_ind) - 1]
                    temp_demand = temp_demand[min(false_ind):max(false_ind) + 1]
                    temp_check = temp_demand < args["capacity_fleet"][f][k]
                    false_ind = list(np.where(temp_check == False)[0])

    vehicles = {j: [x for x in range(1, 1 + vehicle_number_approximation[j])] for j in args["fleets"]}
    indices = {(i, j, k, f) for f in args["fleets"] for i in args["nodes_fleet"][f] for j in args["nodes_fleet"][f]
               for k in vehicles[f] if i != j}
    x = {(i, j, k, f): 0 for i, j, k, f in indices}
    for key in route.keys():
        route_temp = route[key]
        if route_temp:
            for i in range(len(route_temp) - 1):
                x[route_temp[i], route_temp[i + 1], key[1], key[0]] = 1
            x[n + 2 * key[0] - 1, route_temp[0], key[1], key[0]] = 1
            x[route_temp[len(route_temp) - 1], n + 2 * key[0], key[1], key[0]] = 1
        else:
            x[n + 2 * key[0] - 1, n + 2 * key[0], key[1], key[0]] = 1  # start_depot -> end_depot

    # print(vehicle_number_approximation)
    args["vehicles"] = {f: [k+1 for k in range(vehicle_number_approximation[f])] for f in range(1, F+1)}
    args["capacity_fleet"] = {f: [1.] * vehicle_number_approximation[f] for f in range(1, F+1)}

    init_solution = {"x": x, "t": t}
    return CVRPTW(args=args, init_solution=init_solution)


def add_constraints(mdl, x, t, cvrptw):
    """
    Add constraints to Cplex Model.
    """
    n = len(cvrptw.args["customers"])
    F = len(cvrptw.args["fleets"])
    nodes_F = cvrptw.args["nodes_fleet"]
    vehicles = cvrptw.args["vehicles"]
    customers = cvrptw.args["customers"]
    fleets = cvrptw.args["fleets"]
    D = cvrptw.args["demand_operation"]
    Q = cvrptw.args["capacity_fleet"]
    tau = cvrptw.args["travel_time"]
    start_ea = cvrptw.args["start_early"]
    start_la = cvrptw.args["start_late"]
    S_time = cvrptw.args["duration"]

    # 0.each flight will be served by a fleet one time
    mdl.add_constraints(mdl.sum(x[i, j, k, f] for j in nodes_F[f] for k in vehicles[f] if i != j) == 1
                        for i in customers for f in fleets)

    # 1.capacity
    mdl.add_constraints(
        mdl.sum(D[f][i - 1] * mdl.sum(x[i, j, k, f] for j in nodes_F[f] if i != j) for i in customers) <= Q[f][k - 1]
        for f in fleets for k in vehicles[f])

    # 2.start from depot and return to depot
    for f in range(1, 1 + F):
        mdl.add_constraints(
            mdl.sum(x[n + 2 * f - 1, j, k, f] for j in nodes_F[f] if j != (n + 2 * f - 1)) == 1 for k in vehicles[f])
        mdl.add_constraints(
            mdl.sum(x[i, n + f * 2, k, f] for i in nodes_F[f] if i != (n + f * 2)) == 1 for k in vehicles[f])

        mdl.add_constraints(
            mdl.sum(x[n + 2 * f, j, k, f] for j in nodes_F[f] if j != (n + 2 * f)) == 0 for k in vehicles[f])
        mdl.add_constraints(
            mdl.sum(x[i, n + f * 2 - 1, k, f] for i in nodes_F[f] if i != (n + f * 2 - 1)) == 0 for k in vehicles[f])

    # 3.in and out
    for f in range(1, 1 + F):
        mdl.add_constraints((mdl.sum(x[i, h, k, f] for i in nodes_F[f] if i != h) ==
                             mdl.sum(x[h, j, k, f] for j in nodes_F[f] if j != h))
                            for h in customers for k in vehicles[f])

    # 4.TW start
    for f in range(1, 1 + F):
        for i in nodes_F[f]:
            for j in nodes_F[f]:
                if i != j:
                    mdl.add_if_then(sum(x[i, j, k, f] for k in vehicles[f]) >= 1, t[i, f] + tau[i, j, f] <= t[j, f])
    # for f in range(1, 1 + F):
    #     mdl.add_constraints(t[i, f] + tau[i, j, f] - (10 ** 6) * (1 - x[i, j, k, f]) <= t[j, f] for i in nodes_F[f] for j in nodes_F[f] for k in vehicles[f] if i != j)

    mdl.add_constraints(start_ea[i] <= t[i, 1] for i in nodes_F[1])

    # 5.TW end - 10
    mdl.add_constraints(t[i, 10] <= start_la[i] - S_time[10][i - 1] for i in nodes_F[10])

    # 6.precedence
    mdl.add_constraints(t[i, precedence_item[0]] + S_time[precedence_item[0]][i - 1] <= t[i, precedence_item[1]]
                        for i in cvrptw.args["customers"] for precedence_item in cvrptw.args["precedence"][i])

    return mdl


def construct_cplex_model(cvrptw, x_fixed=None):
    start_time = time.time()
    if cvrptw.model is None:
        cvrptw.model = Model("AGH")
        mdl = cvrptw.model

        cur_t = time.time()
        x = mdl.binary_var_dict(list(cvrptw.visit), name='x')
        t = mdl.integer_var_dict(list(cvrptw.time), lb=-60, name='t')

        cvrptw.x = x
        cvrptw.t = t
        # print("Construct x {}, t {} within {}s".format(len(x), len(t), time.time() - cur_t))

        cur_t = time.time()
        mdl = add_constraints(mdl, x, t, cvrptw)
        # print("Construct Deterministic constraints within {}s".format(time.time() - cur_t))
        # print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

        cur_t = time.time()
        mdl.minimize(mdl.sum((cvrptw.args["distance"][i, j]) * x[i, j, k, f] for i, j, k, f in cvrptw.visit))
        # print("Construct Obj within {}s".format(time.time() - cur_t))
    elif cvrptw.lns_constraints is not None:
        mdl = cvrptw.model
        # Remove previous lns_cts
        cur_t = time.time()
        mdl.remove_constraints(cvrptw.lns_constraints)
        # print("Remove {} LNS Constraints within {}s".format(len(cvrptw.lns_constraints), time.time() - cur_t))
        # print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    if x_fixed is not None:
        # Add new lns_cts
        mdl = cvrptw.model
        cur_t = time.time()
        pre_x = cvrptw.visit
        cvrptw.lns_constraints = mdl.add_constraints([cvrptw.x[key] == pre_x[key] for key in x_fixed])  # fix part of variable by add_constraints
        # print("Add {} New LNS Constraints within {}s".format(len(cvrptw.lns_constraints), time.time() - cur_t))
        # print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    # print("Construct Cplex model Within {}s".format(time.time() - start_time))
    cvrptw.construct_time = cvrptw.construct_time + time.time() - start_time

    return mdl


def random_destroy(cvrptw, degree_of_destruction=0.5, mipgap=0.1, timelimit=10):
    """
    Random Optimize parts of decision variables in cplex model.
    Refer to https://arxiv.org/pdf/2004.00422.pdf
    """
    # Choose decision variables to fix
    x_len = len(cvrptw.visit)
    x_fixed = random.sample(list(cvrptw.visit), int((1 - degree_of_destruction) * x_len))

    # Construct model
    mdl = construct_cplex_model(cvrptw, x_fixed)

    # Solve within timelimit
    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    mdl.parameters.advance = 0  # Do not use advanced start information, start the new solve from scratch
    solution = mdl.solve(log_output=False)
    mdl.get_solve_status()

    return solution


def solve_instance(fleet_info, distance_dict, val_dataset, opts):
    cost, batch_size = [], 1
    assert batch_size == 1, "Can only solve one by one!"
    for input in tqdm(DataLoader(val_dataset, batch_size=batch_size, shuffle=False), disable=opts.no_progress_bar):
        loc = input["loc"][0]  # [graph_size]
        arrival = input["arrival"][0]
        departure = input["departure"][0]
        type = input["type"][0]
        demand = input["demand"][0]  # [fleet_size, graph_size]
        fleet_size, graph_size = demand.size()
        initial_vehicle = {20: 15, 50: 20, 100: 25, 200: 40, 300: 50}.get(opts.graph_size)

        customers = [j for j in range(1, graph_size+1)]
        fleets = [j for j in range(1, fleet_size+1)]
        nodes = customers + [j for j in range(graph_size+1, graph_size+1+fleet_size*2)]
        start_ea = {i: -60 for i in nodes}
        start_la = {i: 10**4 for i in nodes}
        for i in customers:
            start_ea[i] = arrival[i-1].item()
            start_la[i] = departure[i-1].item()
        nodes_F = {}
        for f in range(fleet_size):
            nodes_F[f+1] = customers + [graph_size+1+2*f, graph_size+2+2*f]
        vehicles = {f: [j for j in range(1, initial_vehicle+1)] for f in range(1, fleet_size+1)}
        distance = {(i, j): 0 for i in nodes for j in nodes}
        for i in nodes:
            id_i = loc[i-1].item() if i in customers else 0
            for j in nodes:
                id_j = loc[j-1].item() if j in customers else 0
                distance[(i, j)] = distance_dict[(id_i, id_j)]
        D = {f: demand[f-1].tolist() for f in range(1, fleet_size+1)}
        Q = {f: [1.] * len(vehicles[f]) for f in range(1, fleet_size+1)}
        flight_type = {i: type[i-1].item() for i in customers}
        S_time = {}
        for f in range(1, fleet_size+1):
            duration = []
            for i in customers:
                duration.append(fleet_info["duration"][f][flight_type[i]])
            duration += [0.0] * 2 * fleet_size
            S_time[f] = duration
        tau = {(i, j, f): S_time[f][i-1] + distance[i, j] / 110.0 for i in nodes for j in nodes for f in fleets if i != j}
        # hardcoded based on fleet_info["precedence"]
        prec = [[1, 2], [1, 4], [1, 8], [2, 3], [2, 5], [2, 7], [2, 9], [4, 3], [4, 5], [4, 7], [4, 9], [8, 3], [8, 5],
                [8, 7], [8, 9], [3, 6], [5, 6], [7, 6], [9, 6], [6, 10]]
        precedence = {i: prec for i in customers}

        args = {"customers": customers,
                "fleets": fleets,
                "start_early": start_ea,
                "start_late": start_la,
                "nodes": nodes,
                "nodes_fleet": nodes_F,
                "travel_time": tau,
                "vehicles": vehicles,
                "distance": distance,
                "demand_operation": D,
                "capacity_fleet": Q,
                "duration": S_time,
                "precedence": precedence,
                "flight_type": flight_type}

        if opts.val_method == "cplex":
            timelimit = {20: 1800, 50: 1800, 100: 1800, 200: 3600, 300: 3600}
            cvrptw = CVRPTW(args=args, init_solution=None)
            mdl = construct_cplex_model(cvrptw)
            mdl.parameters.timelimit = timelimit[opts.graph_size]
            solution = mdl.solve(log_output=False)
            mdl.get_solve_status()
            if solution is not None:
                print(solution.objective_value)
                cost.append(solution.objective_value)
            else:
                cost.append(None)
        elif opts.val_method == "lns":
            total_tl = {20: 1800, 50: 1800, 100: 1800, 200: 3600, 300: 3600}
            timelimit = {20: 20, 50: 60, 100: 120, 200: 300, 300: 600}
            start_t = time.time()
            cvrptw = get_initial_sol(args)
            best_obj = cvrptw.obj
            print(">> initial sol: {}".format(cvrptw.obj))
            while time.time()-start_t < total_tl[opts.graph_size]:
                solution = random_destroy(cvrptw, degree_of_destruction=0.5, mipgap=0.1, timelimit=timelimit[opts.graph_size])
                if solution is not None and solution.objective_value < cvrptw.obj:
                    visit = {k: int(cvrptw.x[k].solution_value) for k in cvrptw.visit.keys()}
                    t1 = {k: cvrptw.t[k].solution_value for k in cvrptw.time.keys()}
                    cvrptw.set_solution(visit, t1)
                    best_obj = cvrptw.obj if cvrptw.obj < best_obj else best_obj
            print(">> LNS sol: {}".format(best_obj))
            cost.append(best_obj)
        elif opts.val_method == "lns_sa":
            total_tl = {20: 1800, 50: 1800, 100: 1800, 200: 3600, 300: 3600}
            timelimit = {20: 20, 50: 60, 100: 120, 200: 300, 300: 600}
            start_t, T, iteration = time.time(), 200, 0
            cvrptw = get_initial_sol(args)
            best_obj = cvrptw.obj
            print(">> initial sol: {}".format(cvrptw.obj))
            while time.time() - start_t < total_tl[opts.graph_size]:
                solution = random_destroy(cvrptw, degree_of_destruction=0.5, mipgap=0.1, timelimit=timelimit[opts.graph_size])
                iteration += 1
                T = T * 0.95 if iteration % 10 == 0 else T
                if solution is not None:
                    if solution.objective_value < cvrptw.obj:
                        visit = {k: int(cvrptw.x[k].solution_value) for k in cvrptw.visit.keys()}
                        t1 = {k: cvrptw.t[k].solution_value for k in cvrptw.time.keys()}
                        cvrptw.set_solution(visit, t1)
                        best_obj = cvrptw.obj if cvrptw.obj < best_obj else best_obj
                    else:
                        delta_cost = solution.objective_value - cvrptw.obj
                        ran_accept = np.random.uniform(0, 1)
                        criteria = np.e ** (-delta_cost / T)  # Metropolis criteria
                        if ran_accept <= criteria:
                            visit = {k: int(cvrptw.x[k].solution_value) for k in cvrptw.visit.keys()}
                            t1 = {k: cvrptw.t[k].solution_value for k in cvrptw.time.keys()}
                            cvrptw.set_solution(visit, t1)
            print(">> LNS sol: {}".format(best_obj))
            cost.append(best_obj)
    print(cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="./data/agh/agh20_validation_seed4321.pkl", help="Filename of the dataset to load")
    parser.add_argument("--problem", type=str, default='agh', help="only support airport ground handling in this code")
    parser.add_argument('--graph_size', type=int, default=20, help="Sizes of problem instances (20, 50, 100, 200, 300)")
    parser.add_argument('--val_method', type=str, default='cplex', choices=['cplex', 'lns', 'lns_sa'])
    parser.add_argument('--val_size', type=int, default=1000, help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0, help='Offset where to start in dataset (default 0)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    opts = parser.parse_args()
    pp.pprint(vars(opts))

    # Set the random seed
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    fleet_info_path, distance_path = 'problems/agh/fleet_info.pkl', 'problems/agh/distance.pkl'
    with open(fleet_info_path, 'rb') as f:
        fleet_info = pickle.load(f)
    with open(distance_path, 'rb') as f:
        distance_dict = pickle.load(f)

    problem = load_problem(opts.problem)
    val_dataset = problem.make_dataset(filename=opts.filename, num_samples=opts.val_size, offset=opts.offset)

    print('Validating dataset: {}'.format(opts.filename))
    start_time = time.time()
    solve_instance(fleet_info, distance_dict, val_dataset, opts)
    print(">> End of validation within {:.2f}s".format(time.time() - start_time))
