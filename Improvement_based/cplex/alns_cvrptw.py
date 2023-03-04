#!/usr/bin/env python

import copy
import time
import random
import json
import multiprocessing
from scipy import spatial
from docplex.mp.model import Model
from cplex.callbacks import *
from docplex.mp.model_reader import *

# random.seed = 2020

class CVRPTW(object):
    """
    Solution class for the CVRPTW problem. It has two data members, nodes, and edges.
    nodes is a list of node tuples: (id, coords). The edges data member, then, is
    a mapping from each node to their only outgoing node.
    """

    def __init__(self, args={}, init_solution=None, learning=True):
        """
        args = {"customers": customers, - [1,2,...,30]
            "fleets": fleets, - [1,2,...,10]
            "start_early": start_ea, - {1:19800,2:19620,...,50:0}
            "start_late": start_la, - {1:21600,2:21600,...,50:100000000}
            "nodes": nodes, - [1,2,...,50]
            "nodes_fleet": nodes_F, - {1:[1,2,...,30,31,32], 2: [1,2,...,30,33,34], 10: [1,2,...,30,49,50]}
            "travel_time": tau, duration(task on i) + travel time(from i to j) (i,j,f) - {(1,2,1):136.09636363636363,(1,2,2):...,(1,50,10):}
            "degree_of_destruction": 0.25,
            "vehicles": vehicles, - {1:[1,2,...,20],2:[1,2,...,20],...,10:[1,2,...,20]}
            "distance": distance, - {(1,1):0,(1,2):282.96999999999997,...,(50,50):0}
            "demand_operation": D, - {1:[0,...(30),0],1:[],...,10:[]}
            "capacity_fleet": Q, - {1:[capacity, repeat*20],...,10[]}
            "duration": S_time, - {1:[0,...(30+20),0],2:[],...,10:[]}
            "precedence": precedence} - {1:[[1, 2],[1, 3],[9, 10]],2:[],...,30:[]
            "flight_type": flight_type - {1: 1, 2: 3, 3: 3, 4: 3, ..., 20: 2}
        }
        init_solution = {"x": x,
            "t": t,
            "route": route
        }
        """
        super(CVRPTW).__init__()
        self.args = args
        self.obj = 10 ** 8
        self.model = None
        self.lns_constraints = None
        self.construct_time = 0
        self.learning = learning
        self.accepted = []
        self.vehicle_fleet = [(k, f) for f in self.args["fleets"] for k in self.args["vehicles"][f]]
        if init_solution is not None:
            self.route = init_solution["route"]
            self.set_solution(init_solution["x"], init_solution["t"])
        else:
            # decision variable
            self.visit = {(i, j, k, f): 0 for f in self.args["fleets"] for i in self.args["nodes_fleet"][f] for j in self.args["nodes_fleet"][f] for k in self.args["vehicles"][f] if i != j}
            self.time = {(i, f): 0.0 + args["start_early"][i] for f in self.args["fleets"] for i in self.args["nodes_fleet"][f]}  # the start time of operation f at pos i
            # recorded variable
            self.route = {(f, k): [] for f in self.args["fleets"] for k in self.args["vehicles"][f]}
            for f in self.args["fleets"]:
                self.time[len(self.args["customers"]) + f * 2 - 1, f] = 0
                self.time[len(self.args["customers"]) + f * 2, f] = 10 ** 8

        if learning:
            # Data used for learning method.
            self.id_map = {}  # (i, j, k, f): id
            self.veh_map = {}  # (k, f) : id
            self.data = {"c": [], "x": [], "t": [], "k_f": [], "e": [], "label": []}
            # Construct id_map, veh_map
            count = 0
            for f in self.args["fleets"]:
                for k in self.args["vehicles"][f]:
                    for i in self.args["nodes_fleet"][f]:
                        for j in self.args["nodes_fleet"][f]:
                            if i != j:
                                self.id_map[i, j, k, f] = count
                                count += 1
            for f in self.args["fleets"]:
                for i in self.args["nodes_fleet"][f]:
                    self.id_map[i, f] = count
                    count += 1
            assert len(self.id_map) == (len(self.visit) + len(self.time))
            count = 0
            for f in self.args["fleets"]:
                for k in self.args["vehicles"][f]:
                    self.veh_map[k, f] = count
                    count += 1

    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        """
        TODO: w1 * total_distance + w2 * vehicle_nums
        The objective function is simply the sum of all individual edge lengths,
        using the rounded Euclidean norm.
        """
        n = len(self.args["customers"])
        obj = sum(self.args["distance"][i, j] * self.visit[i, j, k, f] for i, j, k, f in self.visit) \
               - sum(self.visit[n + 2 * f - 1, n + 2 * f, k, f] for f in self.args["fleets"] for k in self.args["vehicles"][f])
        return int(obj)

    def get_solution(self):
        return {"x": self.visit, "t": self.time}

    def set_solution(self, visit, t):
        self.visit = visit
        self.time = t
        self.obj = self.objective()

    def is_valid(self):
        return is_all_visited(self) and constraints_depot(self) and constraints_in_out(self) \
               and constraints_capacity(self) and constraints_precedence(self) and constraints_tw_start(self)


def is_all_visited(cvrptw):
    """
    each flight will be served by a fleet one time.
    mdl.add_constraints(mdl.sum(x[i, j, k, f] for j in nodes_F[f] for k in vehicles[f] if i != j) == 1 for i in customers for f in fleets)
    """
    for f in cvrptw.args["fleets"]:
        for i in cvrptw.args["customers"]:
            sum = 0
            for k in cvrptw.args["vehicles"][f]:
                for j in cvrptw.args["nodes_fleet"][f]:
                    if i != j:
                        sum += cvrptw.visit[(i, j, k, f)]
            if sum != 1:
                print("is_all_visited Failed")
                return False
    return True


def constraints_depot(cvrptw):
    """
    start from depot and return to depot. There's a case, a car isn't used, so visit(start_depot, end_depot, k, f) = 1
    mdl.add_constraints(
        mdl.sum(x[i, n + f * 2, k, f] for i in nodes_F[f] if i != (n + f * 2)) == 1 for k in vehicles[f]) - sum1
    mdl.add_constraints(
        mdl.sum(x[i, n + f * 2 - 1, k, f] for i in nodes_F[f] if i != (n + f * 2 - 1)) == 0 for k in vehicles[f]) - sum2
    mdl.add_constraints(
        mdl.sum(x[n + 2 * f - 1, j, k, f] for j in nodes_F[f] if j != (n + 2 * f - 1)) == 1 for k in vehicles[f]) - sum3
    mdl.add_constraints(
        mdl.sum(x[n + 2 * f, j, k, f] for j in nodes_F[f] if j != (n + 2 * f)) == 0 for k in vehicles[f]) - sum4
    """
    for f in cvrptw.args["fleets"]:
        start_depot = len(cvrptw.args["customers"]) + 2 * f - 1
        end_depot = len(cvrptw.args["customers"]) + 2 * f
        for k in cvrptw.args["vehicles"][f]:
            sum1, sum2, sum3, sum4 = 0, 0, 0, 0
            for i in cvrptw.args["nodes_fleet"][f]:
                if i != end_depot:
                    sum1 += cvrptw.visit[(i, end_depot, k, f)]
                    sum4 += cvrptw.visit[(end_depot, i, k, f)]
                if i != start_depot:
                    sum2 += cvrptw.visit[(i, start_depot, k, f)]
                    sum3 += cvrptw.visit[(start_depot, i, k, f)]
            if sum1 != 1 or sum2 != 0 or sum3 != 1 or sum4 != 0:
                print("constraints_depot Failed")
                return False
    return True


def constraints_in_out(cvrptw):
    """
    In and Out.
    mdl.add_constraints((mdl.sum(x[i, h, k, f] for i in nodes_F[f] if i != h) == mdl.sum(x[h, j, k, f] for j in nodes_F[f] if j != h)) for h in customers for k in vehicles[f])
    """
    for f in cvrptw.args["fleets"]:
        for h in cvrptw.args["customers"]:
            for k in cvrptw.args["vehicles"][f]:
                sum_in, sum_out = 0, 0
                for i in cvrptw.args["nodes_fleet"][f]:
                    if i != h:
                        sum_in += cvrptw.visit[(i, h, k, f)]
                        sum_out += cvrptw.visit[(h, i, k, f)]
                # here, we don't need to consider whether sum_in/sum_out > 1 or not, is_all_visited will make sure it.
                if sum_in != sum_out:
                    print("constraints_in_out Failed")
                    return False
    return True


def constraints_capacity(cvrptw):
    """
    each vehicle loads should not exceed its capacity. In our case, the capacity is increasing through its route.
    mdl.add_constraints(mdl.sum(D[f][i - 1] * mdl.sum(x[i, j, k, f] for j in nodes_F[f] if i != j) for i in customers) <= Q[f][k - 1] for f in fleets for k in vehicles[f])
    """
    for f in cvrptw.args["fleets"]:
        for k in cvrptw.args["vehicles"][f]:
            cur_capacity = 0
            for i in cvrptw.args["customers"]:
                cur_demand = cvrptw.args["demand_operation"][f][i-1]
                for j in cvrptw.args["nodes_fleet"][f]:
                    if i != j and cvrptw.visit[(i, j, k, f)] == 1:
                        cur_capacity += cur_demand
            if cur_capacity > cvrptw.args["capacity_fleet"][f][k-1]:
                print("constraints_capacity Failed")
                return False
    return True


def constraints_tw_start(cvrptw):
    """
    Time window constraints.
    mdl.add_if_then(sum(x[i, j, k, f] for k in vehicles[f]) >= 1, t[i, f] + tau[i, j, f] <= t[j, f])
    mdl.add_constraints(start_ea[i] <= t[i, 1] for i in nodes_F[1])
    mdl.add_constraints(t[i, 10] <= start_la[i] - S_time[10][i - 1] for i in nodes_F[10])
    """
    for f in cvrptw.args["fleets"]:
        for i in cvrptw.args["nodes_fleet"][f]:
            for j in cvrptw.args["nodes_fleet"][f]:
                if i != j:
                    visited = False
                    for k in cvrptw.args["vehicles"][f]:
                        if cvrptw.visit[(i, j, k, f)] == 1:
                            visited = True
                            break
                    if visited and cvrptw.time[i, f] + int(cvrptw.args["travel_time"][i, j, f]) > cvrptw.time[j, f]:
                        print(i, j, f)
                        print(cvrptw.time[i, f], cvrptw.args["travel_time"][i, j, f], cvrptw.time[j, f])
                        print("constraints_tw_start Failed1")
                        return False

    # Task 1 is the first operation, task 10 is the last operation
    first_oper, last_oper = 1, len(cvrptw.args["fleets"])
    for i in cvrptw.args["nodes_fleet"][first_oper]:
        if cvrptw.args["start_early"][i] > cvrptw.time[i, first_oper]:
            print("constraints_tw_start Failed2")
            return False
    for i in cvrptw.args["nodes_fleet"][last_oper]:
        if cvrptw.args["start_late"][i] < cvrptw.time[i, last_oper] + cvrptw.args["duration"][last_oper][i-1]:
            print("constraints_tw_start Failed3")
            return False
    return True


def constraints_precedence(cvrptw):
    """
    Precedence constraints.
    Note: Operations can be executed simultaneously
    mdl.add_constraints((t[i, retrieve[yx[1][1][1:len(yx[1][1])][j]]] + S_time[retrieve[yx[1][1][1:len(yx[1][1])][j]]][i - 1])
                            <= t[i, retrieve[yx[1][2][1:len(yx[1][2])][j]]] for j in ind)
    """
    for i in cvrptw.args["customers"]:
        for precedence_item in cvrptw.args["precedence"][i]:
            if cvrptw.time[i, precedence_item[0]] + cvrptw.args["duration"][precedence_item[0]][i-1] > cvrptw.time[i, precedence_item[1]]:
                print("customer {}".format(i))
                print(cvrptw.time[i, precedence_item[0]], cvrptw.args["duration"][precedence_item[0]][i-1], cvrptw.time[i, precedence_item[1]])
                print("constraints_precedence Failed")
                return False
    return True


def add_constraints_learning(mdl, x, t, cvrptw):
    """
    Add constraints to Cplex Model.
    """
    print("[*] add_constraints_learning")
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

    # Extract c's and e's feature
    # self.data = {"c": [], "v": [], "e": [], "label": []}
    # dim - c: 4 e: 3 v: 5
    # 10: == | 01: <=
    count = 0
    obj_vec = [0] * len(cvrptw.visit)
    for i, j, k, f in cvrptw.visit:
        obj_vec[cvrptw.id_map[i, j, k, f]] = cvrptw.args["distance"][i, j]

    # 0.each flight will be served by a fleet one time
    mdl.add_constraints(mdl.sum(x[i, j, k, f] for j in nodes_F[f] for k in vehicles[f] if i != j) == 1 for i in customers for f in fleets)

    for i in customers:
        for f in fleets:
            c_vec = [0] * len(cvrptw.visit)
            for j in nodes_F[f]:
                for k in vehicles[f]:
                    if i != j:
                        c_vec[cvrptw.id_map[i, j, k, f]] = 1
                        cvrptw.data["e"].append([count, cvrptw.id_map[i, j, k, f], 1])
            cos_sim = 1 - spatial.distance.cosine(obj_vec, c_vec)
            cvrptw.data["c"].append([cos_sim, 1, 1, 0])
            count += 1

    print("[1]: constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    # 1.capacity
    mdl.add_constraints(
        mdl.sum(D[f][i - 1] * mdl.sum(x[i, j, k, f] for j in nodes_F[f] if i != j) for i in customers) <= Q[f][k - 1]
        for f in fleets for k in vehicles[f])

    for f in fleets:
        for k in vehicles[f]:
            b = 1 if Q[f][k - 1] != 0 else 0
            c_vec = [0] * len(cvrptw.visit)
            for i in customers:
                for j in nodes_F[f]:
                    if i != j:
                        if b != 0:
                            c_vec[cvrptw.id_map[i, j, k, f]] = D[f][i - 1]/Q[f][k - 1]
                            cvrptw.data["e"].append([count, cvrptw.id_map[i, j, k, f], D[f][i - 1]/Q[f][k - 1]])
                        else:
                            c_vec[cvrptw.id_map[i, j, k, f]] = D[f][i - 1]
                            cvrptw.data["e"].append([count, cvrptw.id_map[i, j, k, f], D[f][i - 1]])
            cos_sim = 1 - spatial.distance.cosine(obj_vec, c_vec) if c_vec != [0] * len(cvrptw.visit) else 0
            cvrptw.data["c"].append([cos_sim, b, 0, 1])
            count += 1

    print("[2]: constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

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

    for f in range(1, 1 + F):
        for k in vehicles[f]:
            c_vec = [0] * len(cvrptw.visit)
            for j in nodes_F[f]:
                if j != (n + 2 * f - 1):
                    c_vec[cvrptw.id_map[n + 2 * f - 1, j, k, f]] = 1
                    cvrptw.data["e"].append([count, cvrptw.id_map[n + 2 * f - 1, j, k, f], 1])
            cos_sim = 1 - spatial.distance.cosine(obj_vec, c_vec)
            cvrptw.data["c"].append([cos_sim, 1, 1, 0])
            count += 1
        for k in vehicles[f]:
            c_vec = [0] * len(cvrptw.visit)
            for i in nodes_F[f]:
                if i != (n + f * 2):
                    c_vec[cvrptw.id_map[i, n + f * 2, k, f]] = 1
                    cvrptw.data["e"].append([count, cvrptw.id_map[i, n + f * 2, k, f], 1])
            cos_sim = 1 - spatial.distance.cosine(obj_vec, c_vec)
            cvrptw.data["c"].append([cos_sim, 1, 1, 0])
            count += 1
        for k in vehicles[f]:
            c_vec = [0] * len(cvrptw.visit)
            for j in nodes_F[f]:
                if j != (n + 2 * f):
                    c_vec[cvrptw.id_map[n + 2 * f, j, k, f]] = 1
                    cvrptw.data["e"].append([count, cvrptw.id_map[n + 2 * f, j, k, f], 1])
            cos_sim = 1 - spatial.distance.cosine(obj_vec, c_vec)
            cvrptw.data["c"].append([cos_sim, 0, 1, 0])
            count += 1
        for k in vehicles[f]:
            c_vec = [0] * len(cvrptw.visit)
            for i in nodes_F[f]:
                if i != (n + f * 2 - 1):
                    c_vec[cvrptw.id_map[i, n + f * 2 - 1, k, f]] = 1
                    cvrptw.data["e"].append([count, cvrptw.id_map[i, n + f * 2 - 1, k, f], 1])
            cos_sim = 1 - spatial.distance.cosine(obj_vec, c_vec)
            cvrptw.data["c"].append([cos_sim, 0, 1, 0])
            count += 1

    print("[3]: constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    # 3.in and out
    for f in range(1, 1 + F):
        mdl.add_constraints((mdl.sum(x[i, h, k, f] for i in nodes_F[f] if i != h) == mdl.sum(x[h, j, k, f] for j in nodes_F[f] if j != h))
                            for h in customers for k in vehicles[f])
    for f in range(1, 1 + F):
        for k in vehicles[f]:
            for h in customers:
                c_vec = [0] * len(cvrptw.visit)
                for i in nodes_F[f]:
                    if i != h:
                        c_vec[cvrptw.id_map[i, h, k, f]] = 1
                        c_vec[cvrptw.id_map[h, i, k, f]] = -1
                        cvrptw.data["e"].append([count, cvrptw.id_map[i, h, k, f], 1])
                        cvrptw.data["e"].append([count, cvrptw.id_map[h, i, k, f], -1])
                cos_sim = 1 - spatial.distance.cosine(obj_vec, c_vec)
                cvrptw.data["c"].append([cos_sim, 0, 1, 0])
                count += 1

    print("[4]: constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    # 4.TW
    for f in range(1, 1 + F):
        for i in nodes_F[f]:
            for j in nodes_F[f]:
                if i != j:
                    mdl.add_if_then(sum(x[i, j, k, f] for k in vehicles[f]) >= 1, t[i, f] + tau[i, j, f] <= t[j, f])
    # for f in range(1, 1 + F):
    #     mdl.add_constraints(t[i, f] + tau[i, j, f] - (10 ** 5) * (1 - x[i, j, k, f]) <= t[j, f] for i in nodes_F[f] for j in nodes_F[f] for k in vehicles[f] if i != j)  # too many constraints - i*j*k*f
    #     mdl.add_constraints(t[i, f] + tau[i, j, f] - (10 ** 5) * (1 - sum(x[i, j, k, f] for k in vehicles[f])) <= t[j, f] for i in nodes_F[f] for j in nodes_F[f] if i != j)  # i*j*f

    for f in range(1, 1 + F):
        for i in nodes_F[f]:
            for j in nodes_F[f]:
                if i != j:
                    c_vec = [0] * len(cvrptw.visit)
                    for k in vehicles[f]:
                        c_vec[cvrptw.id_map[i, j, k, f]] = 10**5/(10**5-tau[i, j, f])
                        cvrptw.data["e"].append([count, cvrptw.id_map[i, j, k, f], 10 ** 5/(10 ** 5 - tau[i, j, f])])
                    cvrptw.data["e"].append([count, cvrptw.id_map[i, f], 1 / (10 ** 5 - tau[i, j, f])])
                    cvrptw.data["e"].append([count, cvrptw.id_map[j, f], -1 / (10 ** 5 - tau[i, j, f])])
                    cos_sim = 1 - spatial.distance.cosine(obj_vec, c_vec)
                    cvrptw.data["c"].append([cos_sim, 1, 0, 1])
                    count += 1

    print("[5]: constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    # 5. TW start and end (1 & 10)
    mdl.add_constraints(start_ea[i] <= t[i, 1] for i in nodes_F[1])
    mdl.add_constraints(t[i, 10] <= start_la[i] - S_time[10][i - 1] for i in nodes_F[10])

    for i in nodes_F[1]:
        b = 1 if start_ea[i] != 0 else 0
        if b != 0:
            cvrptw.data["e"].append([count, cvrptw.id_map[i, 1], 1 / start_ea[i]])
        else:
            cvrptw.data["e"].append([count, cvrptw.id_map[i, 1], 1])
        cvrptw.data["c"].append([0, b, 0, 1])
        count += 1

    for i in nodes_F[10]:
        b = 1 if (start_la[i] - S_time[10][i - 1]) != 0 else 0
        if b != 0:
            cvrptw.data["e"].append([count, cvrptw.id_map[i, 10], 1 / (start_la[i] - S_time[10][i - 1])])
        else:
            cvrptw.data["e"].append([count, cvrptw.id_map[i, 10], 1])
        cvrptw.data["c"].append([0, b, 0, 1])
        count += 1

    print("[6]: constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    # 6.precedence
    mdl.add_constraints(t[i, precedence_item[0]] + S_time[precedence_item[0]][i - 1] <= t[i, precedence_item[1]]
                        for i in cvrptw.args["customers"] for precedence_item in cvrptw.args["precedence"][i])

    for i in cvrptw.args["customers"]:
        for precedence_item in cvrptw.args["precedence"][i]:
            b = 1 if S_time[precedence_item[0]][i - 1] != 0 else 0
            if b != 0:
                cvrptw.data["e"].append([count, cvrptw.id_map[i, precedence_item[0]], 1 / S_time[precedence_item[0]][i - 1]])
                cvrptw.data["e"].append([count, cvrptw.id_map[i, precedence_item[1]], -1 / S_time[precedence_item[0]][i - 1]])
            else:
                cvrptw.data["e"].append([count, cvrptw.id_map[i, precedence_item[0]], 1])
                cvrptw.data["e"].append([count, cvrptw.id_map[i, precedence_item[1]], -1])
            cvrptw.data["c"].append([0, b, 0, 1])
            count += 1

    print("[7]: constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))
    print(len(cvrptw.data["c"]), len(cvrptw.data["e"]))

    return mdl


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


class myModel(Model):
    """
    Overwrite Docplex API.
    Not removed If_then constraints, which are removed inaccurately by remove_constraints API.
    """
    def __init__(self, name=None, context=None, **kwargs):
        super(myModel, self).__init__(name=name, context=context, **kwargs)

    def _remove_constraints_internal(self, doomed):
        # INTERNAL
        self_cts_by_name = self._cts_by_name
        doomed_indices = set()
        for d in doomed:
            if self_cts_by_name:
                dname = d.get_name()
                if dname:
                    del self_cts_by_name[dname]
            if d._index >= 0:
                doomed_indices.add(d._index)
        # update container
        # print(len(self._Model__allcts))
        # # c = LinearConstraint()
        # # c.is_linear()
        # # c.set_index()
        # print(doomed_indices)
        # test = []
        # for c in self._Model__allcts:
        #     if c._index == 2153:
        #         print(c)
        #         print(c.is_linear())
        #         print(type(c))
        #     test.append(c._index)
        # print(test)
        # print(len(test))
        self._Model__allcts = [c for c in self._Model__allcts if c._index not in doomed_indices or not c.is_linear()]
        # print(len(self._Model__allcts))
        # TODO: handle reindexing
        doomed_scopes = set(c._get_index_scope() for c in doomed)
        for ds in doomed_scopes:
            ds.reindex_all(self._Model__engine)
            ds.update_indices()
        for d in doomed:
            d.notify_deleted()


def by_vehicle_destroy(cvrptw, degree_of_destruction=0.1, mipgap=0.1, timelimit=300, start=0, shuffle="DISTANCE_DESCEND"):
    """
    Define a decomposition of the set X(Integer Variable) as a disjoint union X1 ∪ X2 ∪ ··· ∪ Xk.
    Decomposition by vehicle - disjoint!
    shuffle == "NONE": if_shuffle is False
    shuffle == "RANDOM": choose (k, f) randomly, if_shuffle is True
    shuffle == "RANDOM_NO_DISJOINT": choose (k, f) randomly, if_shuffle is True, not disjoint between adjacent iters.
    shuffle == "DISTANCE_DESCEND": choose (k, f) by length of vehicle's route in Descending Order
    shuffle == "DISTANCE_DESCEND_NO_DISJOINT": shuffle every iter, not disjoint between adjacent iters
    shuffle == "DISTANCE_ASCEND": choose (k, f) by length of vehicle's route in Ascending Order
    shuffle == "BOTH_ENDS": choose (k, f) from both ends of the sorted length list of (k, f)
    """
    destroy_num = int(len(cvrptw.vehicle_fleet) * degree_of_destruction)
    # decrease destroy_degree for the top k longest vehicle
    # if start <= len(cvrptw.vehicle_fleet) * 0.05 or (start + destroy_num)/len(cvrptw.vehicle_fleet) > 1.05:
    #     print("[!] Decrease destroy_degree.")
    #     destroy_num = int(destroy_num * 0.65)
    if shuffle == "BOTH_ENDS":
        weight = 0.5
        destroy_num = int(destroy_num/2)
        end = start + destroy_num if start + destroy_num <= len(cvrptw.vehicle_fleet) else start + destroy_num - len(cvrptw.vehicle_fleet)
        destroy_id = cvrptw.vehicle_fleet_descend[start: end] + cvrptw.vehicle_fleet_ascend[start: end] if start < end else cvrptw.vehicle_fleet_descend[start:] + cvrptw.vehicle_fleet_ascend[start:]
        if start >= end:
            shuffle_vehicle_fleet(cvrptw, shuffle=shuffle)
        destroy_id = set(destroy_id) | set(cvrptw.vehicle_fleet_descend[:end]) | set(cvrptw.vehicle_fleet_ascend[:end]) if start >= end else destroy_id
        destroy_degree = len(destroy_id) / len(cvrptw.vehicle_fleet_descend)
        compensate = degree_of_destruction - destroy_degree
        if compensate >= 0.1:
            print("[!] trigger compensate mechanism: compensate {}".format(compensate))
            tmp = end
            end += int(len(cvrptw.vehicle_fleet) * compensate / 2)
            destroy_id = set(destroy_id) | set(cvrptw.vehicle_fleet_descend[tmp:end]) | set(cvrptw.vehicle_fleet_ascend[tmp:end])
            destroy_degree = len(destroy_id) / len(cvrptw.vehicle_fleet_descend)
        fix_id = set(cvrptw.vehicle_fleet_descend) - set(destroy_id)
        print("Optimize {:.2f}% vars, while Fix {:.2f}% vars".format(destroy_degree * 100, (1 - destroy_degree) * 100))
    elif shuffle == "RANDOM_NO_DISJOINT":
        end = 0
        destroy_id = random.sample(cvrptw.vehicle_fleet, destroy_num)
        fix_id = set(cvrptw.vehicle_fleet) - set(destroy_id)
        destroy_degree = len(destroy_id) / len(cvrptw.vehicle_fleet)
        print("Optimize {:.2f}% vars, while Fix {:.2f}% vars".format(destroy_degree * 100, (1 - destroy_degree) * 100))
    elif shuffle == "DISTANCE_DESCEND_NO_DISJOINT":
        end = 0
        shuffle_vehicle_fleet(cvrptw, shuffle="DISTANCE_DESCEND")
        destroy_id = cvrptw.vehicle_fleet[: destroy_num]
        fix_id = set(cvrptw.vehicle_fleet) - set(destroy_id)
        destroy_degree = len(destroy_id) / len(cvrptw.vehicle_fleet)
        print("Optimize {:.2f}% vars, while Fix {:.2f}% vars".format(destroy_degree * 100, (1 - destroy_degree) * 100))
    else:
        end = start + destroy_num if start + destroy_num <= len(cvrptw.vehicle_fleet) else start + destroy_num - len(cvrptw.vehicle_fleet)
        destroy_id = cvrptw.vehicle_fleet[start: end] if start < end else cvrptw.vehicle_fleet[start:]
        if start >= end:
            shuffle_vehicle_fleet(cvrptw, shuffle=shuffle)
        destroy_id = set(destroy_id) | set(cvrptw.vehicle_fleet[:end]) if start >= end else destroy_id
        destroy_degree = len(destroy_id)/len(cvrptw.vehicle_fleet)
        compensate = degree_of_destruction - destroy_degree
        if compensate >= 0.1:
            print("[!] trigger compensate mechanism: compensate {}".format(compensate))
            tmp = end
            end += int(len(cvrptw.vehicle_fleet) * compensate)
            destroy_id = set(destroy_id) | set(cvrptw.vehicle_fleet[tmp:end])
            destroy_degree = len(destroy_id)/len(cvrptw.vehicle_fleet)
        fix_id = set(cvrptw.vehicle_fleet) - set(destroy_id)
        print("Optimize {:.2f}% vars, while Fix {:.2f}% vars".format(destroy_degree * 100, (1-destroy_degree) * 100))

    print(destroy_id)

    if cvrptw.learning:
        # extract label
        label = [0] * len(cvrptw.veh_map)
        for i in destroy_id:
            label[cvrptw.veh_map[i[0], i[1]]] = 1
        cvrptw.data["label"].append(label)

    if shuffle != "RANDOM" and shuffle != "NONE":
        for j in destroy_id:
            print(cvrptw.vehicle_distance[j], end=" ")
    print("")
    x_fixed = [(i, j, k, f) for i, j, k, f in cvrptw.visit if (k, f) in fix_id]

    # Construct model
    mdl = construct_cplex_model(cvrptw, x_fixed)

    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    mdl.parameters.advance = 0  # Do not use advanced start information, start the new solve from scratch
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()

    return solution, end


def random_destroy(cvrptw, degree_of_destruction=0.1, mipgap=0.1, timelimit=300):
    """
    Random Optimize parts of decision variables in cplex model.
    1. Fix part of variable by add_constraints:
        By setting lb and ub: mdl.integer_var(lb=lb, ub=ub) needs a large amount of time.
        By add_constraints works well.
    2. Construct mdl from scratch every time - Remove a batch of Constraints.
        var - 10s; add_constraints - 50s; extra add_constraints - 10s (destroy-0.8)
        Customer 100: copy.deepcopy(mdl) needs 100s, while constructing from scratch needs 60s.
        File System is also a kind of deepcopy, the time needed to load a model is almost the same as constructing from scratch.
        Remove Constraints (Remove one by one) needs more time even if Customer = 20.
        lazy_constraints will increase the solving time tremendously.
    3. t continuous_var may not satisfy constraints_precedence.
        e.g. 21240.000000000004 + 240.0 > 21480.0
    """
    # Choose decision variables to fix
    x_len = len(cvrptw.visit)
    x_fixed = random.sample(list(cvrptw.visit), int((1 - degree_of_destruction) * x_len))

    # Construct model
    mdl= construct_cplex_model(cvrptw, x_fixed)

    # Solve within timelimit
    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    mdl.parameters.advance = 0  # Do not use advanced start information, start the new solve from scratch
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()

    return solution


def local_branching(cvrptw, mipgap=0.1, timelimit=300, k=100):
    """
    Fischetti, Matteo, and Andrea Lodi. "Local branching." Mathematical programming 98.1-3 (2003): 23-47.
    """
    S_0, S_1 = [], []
    for key, value in cvrptw.visit.items():
        if value == 1:
            S_1.append(key)
        else:
            S_0.append(key)
    print(len(S_0), len(S_1))

    # add local_branching constraint
    mdl = construct_cplex_model(cvrptw, None)
    x, t = cvrptw.x, cvrptw.t

    cur_t = time.time()
    # TODO: remove constraints
    # mdl.add_constraint(sum(1 - x[key] for key in S_1) + sum(x[key1] for key1 in S_0) <= k)
    mdl.add_constraint(sum(x[key1] for key1 in S_1) <= int(k/2))  # 2/k-opt
    print("Construct local_branching constraints within {}s".format(time.time() - cur_t))

    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    mdl.parameters.advance = 0  # Do not use advanced start information, start the new solve from scratch
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()

    return solution


def by_fleet_destroy(cvrptw, fleet=1, mipgap=0.1, timelimit=300):
    """
    Optimize sol by fleets (Related Removal).
    Relatedness + Randomness.
    """
    rand = random.randint(1, 10)
    fleet = (fleet + rand) % 10 + 1
    destroy_id = [(k, f) for f in cvrptw.args["fleets"] for k in cvrptw.args["vehicles"][f] if f == fleet]
    fix_id = set(cvrptw.vehicle_fleet) - set(destroy_id)
    destroy_degree = len(destroy_id)/len(cvrptw.vehicle_fleet)
    print("Optimize {:.2f}% vars, while Fix {:.2f}% vars".format(destroy_degree * 100, (1-destroy_degree) * 100))

    print(destroy_id)
    for j in destroy_id:
        print(cvrptw.vehicle_distance[j], end=" ")
    x_fixed = [(i, j, k, f) for i, j, k, f in cvrptw.visit if (k, f) in fix_id]

    # Construct model
    mdl = construct_cplex_model(cvrptw, x_fixed)

    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    mdl.parameters.advance = 0  # Do not use advanced start information, start the new solve from scratch
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()

    return solution


def by_vehicle_destroy_randomness(cvrptw, degree_of_destruction=0.1, randomness=0.6, mipgap=0.1, timelimit=300, start=0, shuffle="DISTANCE_DESCEND", disjoint=False):
    """
    Vehicle_LNS with Randomness.
    Worst Removal + Relatedness + Randomness
    [!] BOTH_ENDS is not supported.
    """
    destroy_num = int(len(cvrptw.vehicle_fleet) * degree_of_destruction)
    random_num = int(destroy_num * randomness)
    destroy_num -= random_num

    if not disjoint:
        end = 0
        shuffle_vehicle_fleet(cvrptw, shuffle=shuffle)
        destroy_id = set(cvrptw.vehicle_fleet[: destroy_num])
    else:
        if start + destroy_num < len(cvrptw.vehicle_fleet):
            end = start + destroy_num
            destroy_id = set(cvrptw.vehicle_fleet[start: end])
        else:
            end = 0
            destroy_id = set(cvrptw.vehicle_fleet[start:])
            random_num += start + destroy_num - len(cvrptw.vehicle_fleet)
            shuffle_vehicle_fleet(cvrptw, shuffle=shuffle)

    while random_num > 0:
        v = random.sample(cvrptw.vehicle_fleet, 1)
        for vv in v:
            if vv not in destroy_id:
                random_num -= 1
                destroy_id.add(vv)

    destroy_degree = len(destroy_id)/len(cvrptw.vehicle_fleet)
    fix_id = set(cvrptw.vehicle_fleet) - set(destroy_id)
    print("Optimize {:.2f}% vars, while Fix {:.2f}% vars".format(destroy_degree * 100, (1-destroy_degree) * 100))
    print(destroy_id)
    if cvrptw.learning:
        # extract label
        label = [0] * len(cvrptw.veh_map)
        for i in destroy_id:
            label[cvrptw.veh_map[i[0], i[1]]] = 1
        cvrptw.data["label"].append(label)

    if shuffle != "RANDOM" and shuffle != "NONE":
        for j in destroy_id:
            print(cvrptw.vehicle_distance[j], end=" ")
    print("")
    x_fixed = [(i, j, k, f) for i, j, k, f in cvrptw.visit if (k, f) in fix_id]

    # Construct model
    mdl = construct_cplex_model(cvrptw, x_fixed)

    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    mdl.parameters.advance = 0  # Do not use advanced start information, start the new solve from scratch
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()

    return solution, end


def construct_cplex_model(cvrptw, x_fixed, loading_path=None):
    start_time = time.time()
    if cvrptw.model is None:
        cvrptw.model = myModel("CVRPTW_multi_fleet")
        mdl = cvrptw.model

        cur_t = time.time()
        x = mdl.binary_var_dict(list(cvrptw.visit), name='x')
        # solve t from scratch
        t = mdl.integer_var_dict(list(cvrptw.time), lb=-3700, name='t')

        # t = mdl.continuous_var_dict(cvrptw.time, name='t')
        cvrptw.x = x
        cvrptw.t = t
        print("Construct x {}, t {} within {}s".format(len(x), len(t), time.time() - cur_t))

        cur_t = time.time()
        if cvrptw.learning:
            if not loading_path:
                mdl = add_constraints_learning(mdl, x, t, cvrptw)
            else:
                mdl = add_constraints(mdl, x, t, cvrptw)
                with open(loading_path, "r") as f:
                    shared_data = json.load(f)
                cvrptw.data["c"], cvrptw.data["e"] = shared_data["0"]["c"], shared_data["0"]["e"]
        else:
            mdl = add_constraints(mdl, x, t, cvrptw)
        print("Construct Deterministic constraints within {}s".format(time.time() - cur_t))
        print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

        cur_t = time.time()
        n = len(cvrptw.args["customers"])
        mdl.minimize(mdl.sum((cvrptw.args["distance"][i, j]) * x[i, j, k, f] for i, j, k, f in cvrptw.visit)
                     - mdl.sum(x[n + 2 * f - 1, n + 2 * f, k, f] for f in cvrptw.args["fleets"] for k in cvrptw.args["vehicles"][f]))
        print("Construct Obj within {}s".format(time.time() - cur_t))
    # 07/14/2020 for local_branching
    if x_fixed is None:
        return cvrptw.model
    # 07/14/2020 END
    elif cvrptw.lns_constraints is not None:
        mdl = cvrptw.model
        # Remove previous lns_cts
        cur_t = time.time()
        mdl.remove_constraints(cvrptw.lns_constraints)
        print("Remove {} LNS Constraints within {}s".format(len(cvrptw.lns_constraints), time.time() - cur_t))
        print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    # Add new lns_cts
    mdl = cvrptw.model
    cur_t = time.time()
    pre_x = cvrptw.visit
    cvrptw.lns_constraints = mdl.add_constraints([cvrptw.x[key] == pre_x[key] for key in x_fixed])  # fix part of variable by add_constraints
    print("Add {} New LNS Constraints within {}s".format(len(cvrptw.lns_constraints), time.time() - cur_t))
    print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    print("Construct Cplex model Within {}s".format(time.time() - start_time))
    cvrptw.construct_time = cvrptw.construct_time + time.time() - start_time

    return mdl


def shuffle_vehicle_fleet(cvrptw, shuffle="DISTANCE_ASCEND"):
    if shuffle == "RANDOM":
        print("Shuffling vehicle_fleet.")
        random.shuffle(cvrptw.vehicle_fleet)
    else:
        test = time.time()
        cvrptw.vehicle_distance = {(k, f): 0 for f in cvrptw.args["fleets"] for k in cvrptw.args["vehicles"][f]}
        for key, value in cvrptw.visit.items():
            if value != 1:
                continue
            i, j, k, f = key[0], key[1], key[2], key[3]
            cvrptw.vehicle_distance[(k, f)] += cvrptw.args["distance"][(i, j)]

        # clear all elements whose value is 0
        # need to check destroy_id before destroying, some may not exist any more.
        # key_list = []
        # for key, value in cvrptw.vehicle_distance.items():
        #     if value == 0:
        #         key_list.append(key)
        # for key in key_list:
        #     del cvrptw.vehicle_distance[key]

        if shuffle == "DISTANCE_DESCEND":
            # 1. Descending Order
            cvrptw.vehicle_fleet = sorted(cvrptw.vehicle_distance.keys(), key=lambda item: cvrptw.vehicle_distance[item], reverse=True)
            print("Sorting {} vehicle_distance within {}s".format(len(cvrptw.vehicle_distance), time.time() - test))
        if shuffle == "DISTANCE_ASCEND":
            # 2. Ascending Order
            cvrptw.vehicle_fleet = sorted(cvrptw.vehicle_distance.keys(), key=lambda item: cvrptw.vehicle_distance[item], reverse=False)
            print("Sorting {} vehicle_distance within {}s".format(len(cvrptw.vehicle_distance), time.time() - test))
        if shuffle == "BOTH_ENDS":
            # 1. Descending Order
            cvrptw.vehicle_fleet_descend = sorted(cvrptw.vehicle_distance.keys(), key=lambda item: cvrptw.vehicle_distance[item], reverse=True)
            # 2. Ascending Order
            cvrptw.vehicle_fleet_ascend = sorted(cvrptw.vehicle_distance.keys(), key=lambda item: cvrptw.vehicle_distance[item], reverse=False)
            print("Sorting {} vehicle_distance within {}s".format(len(cvrptw.vehicle_distance), time.time() - test))


def by_vehicle_destroy_with_learning(cvrptw, destroy_id, mipgap=0.1, timelimit=300):
    """
    Choose vehicle to destroy by Imitation Learning or Reinforcement Learning.
    """
    assert destroy_id is not None
    print(destroy_id)
    fix_id = set(cvrptw.vehicle_fleet) - set(destroy_id)
    print("")
    x_fixed = [(i, j, k, f) for i, j, k, f in cvrptw.visit if (k, f) in fix_id]

    # Construct model
    mdl = construct_cplex_model(cvrptw, x_fixed)

    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    mdl.parameters.advance = 0  # Do not use advanced start information, start the new solve from scratch
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()

    # remove LNS constraints
    mdl.remove_constraints(cvrptw.lns_constraints)
    cvrptw.lns_constraints = None
    # print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    return solution


if __name__ == '__main__':
    # TODO: 08/03/2020
    #   1. Adjust formulation dynamically.
    mdl = Model('CVRP_multi_fleet')
    a = {(1, 1, 1, 1), (2, 2, 2, 2)}
    x = mdl.binary_var_dict(a, name='x')
    x[(3, 3, 3, 3)] = mdl.binary_var(name='x_{}_{}_{}_{}'.format(3, 3, 3, 3))
    print(x)
    for i in mdl.iter_binary_vars():
        print(i)
