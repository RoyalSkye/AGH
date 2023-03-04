import xlrd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import random
import xml.etree.ElementTree as ET
import math
import sumolib
# from alns_cvrptw import *
from ortools.sat.python import cp_model

import xlwt
from openpyxl.workbook import Workbook as openpyxlWorkbook

import pickle
from utils import *


class Schedule_opr():

    def __init__(self):

        self.book = xlwt.Workbook()

    def sheet(self, lt1, lt2, lt3, lt4, lt5, lt6, lt7, lt8, lt9, lt10):

        sh = self.book.add_sheet('Sheet1')

        col1_name = 'Item No.'
        col2_name = 'Flight No.'
        col3_name = 'Operation'
        col4_name = 'Start Time'
        col5_name = 'End Time'
        col6_name = 'Vehicle Name'
        col7_name = 'From Gate'
        col8_name = 'Departure Time'
        col9_name = 'Arrival Time'
        col10_name = 'To Gate'
        # col11_name = 'Predict Delay'

        n = 0
        sh.write(n, 0, col1_name)
        sh.write(n, 1, col2_name)
        sh.write(n, 2, col3_name)
        sh.write(n, 3, col4_name)
        sh.write(n, 4, col5_name)
        sh.write(n, 5, col6_name)
        sh.write(n, 6, col7_name)
        sh.write(n, 7, col8_name)
        sh.write(n, 8, col9_name)
        sh.write(n, 9, col10_name)
        # sh.write(n, 10, col11_name)

        for m, e1 in enumerate(lt1, n + 1):
            sh.write(m, 0, e1)
        for m, e2 in enumerate(lt2, n + 1):
            sh.write(m, 1, e2)
        for m, e3 in enumerate(lt3, n + 1):
            sh.write(m, 2, e3)
        for m, e4 in enumerate(lt4, n + 1):
            sh.write(m, 3, e4)
        for m, e5 in enumerate(lt5, n + 1):
            sh.write(m, 4, e5)
        for m, e6 in enumerate(lt6, n + 1):
            sh.write(m, 5, e6)
        for m, e7 in enumerate(lt7, n + 1):
            sh.write(m, 6, e7)
        for m, e8 in enumerate(lt8, n + 1):
            sh.write(m, 7, e8)
        for m, e9 in enumerate(lt9, n + 1):
            sh.write(m, 8, e9)
        for m, e10 in enumerate(lt10, n + 1):
            sh.write(m, 9, e10)
        # for m, e11 in enumerate(lt11, n + 1):
        #     sh.write(m, 10, e11)

        return self.book


def excelreader(name):
    dic = {}
    workbook = xlrd.open_workbook(name)
    sheet_count = workbook.nsheets
    for i in range(sheet_count):
        worksheet = workbook.sheet_by_index(i)
        total_rows = worksheet.nrows
        total_cols = worksheet.ncols
        table = list()
        attribute = list()
        for y in range(total_cols):
            for x in range(total_rows):
                attribute.append(worksheet.cell(x, y).value)
            table.append(attribute)
            attribute = []
            x += 1
        dic[i] = table
    return dic, total_rows


def from_excel_ordinal(ordinal, _epoch0=datetime(1899, 12, 31)):
    if ordinal > 59:
        ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!
    return (_epoch0 + timedelta(days=ordinal)).replace(microsecond=0)


def Loadlocation(prjXMLPath):
    prjFolder = os.path.dirname(prjXMLPath)
    tree = ET.parse(prjXMLPath)
    root = tree.getroot()

    # Read all locations and store as a list of ditct
    locationList = []
    for loc in root.find('locations').findall('loc'):
        locationList.append(loc.attrib)
    return locationList


def findedge(gate, data):
    edge_list = []
    for i in range(len(data)):
        if data[i]['name'] == gate:
            edge_list.append(data[i]['inEdgeID'])
            edge_list.append(data[i]['outEdgeID'])
            break
    return edge_list


# Read info from the input files, construct the ILP model,
# and solve it using CPLEX. The result file will be
# generated using the resultFilePath.
def solveProject(prjCfgPath, schFilePath, opnFilePath, netFilePath, scale):
    xx, xy = excelreader(schFilePath)  # schedule_W_35_original.xlsx
    yx, yy = excelreader(opnFilePath)  # operation_W.xlsx

    # customer number
    n = len(xx[0][0]) - 1
    customers = [x for x in range(1, 1 + n)]
    print(customers, '\n')

    rnd = np.random
    rnd.seed(1)

    #  fleet definition

    F = 10  # fleet number
    fleets = [x for x in range(1, 1 + F)]

    # add depot
    nodes = customers
    for i in range(F):
        nodes = nodes + [n + 1 + 2 * i]
        nodes = nodes + [n + 2 + 2 * i]

    nodes_F = {}
    for i in range(F):
        nodes_F[i + 1] = customers + [n + 1 + 2 * i] + [n + 2 + 2 * i]
    # print(nodes)
    # print(nodes_F)

    type_in_schedule = xx[0][5][1:len(xx[0][5])]
    type_in_schedule = [int(x) for x in type_in_schedule]
    demand = yx[0][3][1:len(yx[0][3])]
    demand = [int(x) for x in demand]

    D = {}
    for i in fleets:
        li = list()
        for j in type_in_schedule:
            li.append(demand[10 * (j - 1) + (i - 1)])
        D[i] = li

    Q_of_vehicle = yx[2][2][1:len(yx[2][1])]
    Q_of_vehicle = [int(x) for x in Q_of_vehicle]

    # read the distance from
    net = sumolib.net.readNet(netFilePath)  # airport.net.xml
    ret = xx[0][4][1:len(xx[0][4])] + ['depot'] * 2 * F
    distance_ret = dict(zip(nodes, ret))
    data = Loadlocation(prjCfgPath)
    distance = {
        (i, j): net.getShortestPath(
            net.getEdge(findedge(distance_ret[i], data)[1]), net.getEdge(findedge(distance_ret[j], data)[0]))[1]
        for i in nodes for j in nodes}
    for i in nodes:
        distance[i, i] = 0
    for i in range(F):
        distance[n + 1 + 2 * i, n + 2 + 2 * i] = 0
        distance[n + 2 + 2 * i, n + 1 + 2 * i] = 0

    # with open(distance_path, 'rb') as f:
    #     distance = pickle.load(f)
    for key in distance.keys():
        distance[key] = int(distance[key])

    # Time windows

    # 10 ** 6 --> 10 ** 8 20191030
    I = 10 ** 8

    start_ = xx[0][2][1:len(xx[0][2])]
    end_ = xx[0][3][1:len(xx[0][3])]

    start_ea = {}
    for i in customers:
        string = start_[i - 1]
        tmpDatetime = from_excel_ordinal(string)  # Convert float to datetime
        time = [tmpDatetime.hour, tmpDatetime.minute]
        # minute --> second 20191030
        start_ea[i] = (time[0] * 60 + time[1]) * 60

    start_la = {}
    for i in customers:
        string = end_[i - 1]
        tmpDatetime = from_excel_ordinal(string)  # Convert float to datetime
        time = [tmpDatetime.hour, tmpDatetime.minute]
        # minute --> second 20191030
        start_la[i] = (time[0] * 60 + time[1]) * 60

    for i in range(2 * F):
        start_ea[n + i + 1] = -3600
        start_la[n + i + 1] = I

    duration = yx[0][2][1:len(yx[0][2])]
    q = {}
    for i in fleets:
        li = list()
        for j in type_in_schedule:
            # minute --> second 20191030
            li.append(int(60 * duration[10 * (j - 1) + (i - 1)]))
        for k in range(2 * F):
            li.append(0)
        q[i] = li
    S_time = {j: q[j] for j in fleets}
    # print('S_time', S_time)

    speed = yx[2][3][1:len(yx[2][3])]
    speed = [int(x) for x in speed]
    # minute --> second 20191030
    # tau = {(i, j, f): S_time[f][i - 1] + 60 * distance[i, j] / (speed[f - 1])  # ADJUST SPEED HERE
    #        for i in nodes for j in nodes for f in fleets if i != j}
    tau = {(i, j, f): S_time[f][i - 1] + int(60 * distance[i, j] / (speed[f - 1]))  # ADJUST SPEED HERE
           for i in nodes for j in nodes for f in fleets if i != j}

    # print('start time', start_ea)
    # print('end time', start_la)

    # 200619 initial solution begin

    # variable t
    indices_tau = {(i, f) for f in fleets for i in nodes_F[f]}

    # initial solution
    initial_t = {(i, f): 0 + start_ea[i] for i, f in indices_tau}
    # 1 calculate start time of each fleet according to precedence
    # remove int(max(...)) and add t[n+2*f-1,f]=0, t[n+2*f,f]=I 04/24/2020
    operation_type = yx[2][0][1:len(yx[2][0])]
    retrieve = dict(zip(operation_type, range(1, len(operation_type) + 1)))  # ["IB": 1, ...]
    ty = yx[1][0][1:len(yx[1][0])]
    ty = [int(x) for x in ty]
    for i in customers:
        Ty = type_in_schedule[i - 1]
        ind = [i for i, x in enumerate(ty) if x == Ty]
        for j in ind:
            initial_t[i, retrieve[yx[1][2][1:len(yx[1][2])][j]]] = \
                max(initial_t[i, retrieve[yx[1][2][1:len(yx[1][2])][j]]],
                    initial_t[i, retrieve[yx[1][1][1:len(yx[1][1])][j]]] +
                    S_time[retrieve[yx[1][1][1:len(yx[1][1])][j]]][i - 1])
    for f in fleets:
        initial_t[n + f * 2 - 1, f], initial_t[n + 2 * f, f] = -3600, I

    # calculate the vehicle number according to time and capacity, generate the route at the same time
    # 根据TW 估计车辆的数量的时候 生成route 然后根据capacity的约束调整route (+vehicle) -> 根据precedence来决定哪个任务先执行t
    # 1 time window
    time_approximation = {f: [] for f in fleets}
    vehicle_number_approximation = {f: 0 for f in fleets}
    for f in fleets:
        for i in customers:
            for j in customers:
                if i != j:
                    time_approximation[f].append(tau[i, j, f])
    time_approximation_avg = {f: np.max(time_approximation[f]) for f in fleets}
    # print(' ')
    # print('period duration', time_approximation_avg)
    # print(' ')
    flight_start = list(start_ea.values())[0: n]

    time_vehicle = {f: {i: 0 for i in customers} for f in fleets}
    route = dict()
    for f in fleets:
        vehicle_count = 1
        while 0 in list(time_vehicle[f].values()):
            route[f, vehicle_count] = []
            current_time = 0
            max_time = max([initial_t[j + 1, f] for j in range(n)])
            previous_flight = 0
            for j in range(n):
                if time_vehicle[f][j + 1] == 0:
                    current_time = initial_t[j + 1, f] + S_time[f][j]
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
                        wait = initial_t[j + 1, f] - current_time \
                               - 60 * distance[previous_flight, j + 1] / (speed[f - 1])
                        # if 0 < wait < 10 * 60:
                        if wait > 0 and nearest_flight == 0:
                            nearest_distance = 60 * distance[previous_flight, j + 1] / (speed[f - 1])
                            nearest_flight = j + 1
                        elif 0 < wait < 60 and nearest_distance > 60 * distance[previous_flight, j + 1] / (speed[f - 1]):
                            nearest_distance = 60 * distance[previous_flight, j + 1] / (speed[f - 1])
                            nearest_flight = j + 1
                if nearest_flight != 0:
                    current_time = initial_t[nearest_flight, f] + S_time[f][nearest_flight - 1]
                    route[f, vehicle_count].append(nearest_flight)
                    time_vehicle[f][nearest_flight] = vehicle_count
                    previous_flight = nearest_flight

            vehicle_count += 1
            # for j in range(n):
            #     max_arc_cost = 0
            #     for i in range(j + 1, n):
            #         if time_vehicle[f][i + 1] == 0 and tau[j + 1, i + 1, f] > max_arc_cost:
            #             max_arc_cost = tau[j + 1, i + 1, f]
            #     # if flight_start[j] > current_time and time_vehicle[f][j + 1] == 0:
            #     if t[j + 1, f] > current_time and time_vehicle[f][j + 1] == 0:
            #         time_vehicle[f][j + 1] = vehicle_count
            #         # current_time = flight_start[j] + time_approximation_avg[f]
            #         current_time = t[j + 1, f] + time_approximation_avg[f]
            #         route[f, vehicle_count].append(j + 1)
            #         # current_time = t[j + 1, f] + max_arc_cost
            # vehicle_count += 1
        # for k in range(1, vehicle_count):
        #     for i in range(n):
        #         if time_vehicle[f][i + 1] == k:
        #             route[f, k].append(i + 1)
        vehicle_number_approximation[f] = np.max(list(time_vehicle[f].values()))
    print('vehicle 1', vehicle_number_approximation)
    # print('route', route)
    # 2 capacity
    for f in fleets:
        if Q_of_vehicle[f - 1] != 0:
            for k in range(1, vehicle_number_approximation[f] + 1):
                # print([demand[(type_in_schedule[i - 1] - 1) * F + f - 1] for i in route[f, k]])
                temp_demand = np.cumsum([demand[(type_in_schedule[i - 1] - 1) * F + f - 1] for i in route[f, k]])
                temp_route = route[f, k]
                temp_check = temp_demand < Q_of_vehicle[f - 1]
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
                    temp_check = temp_demand < Q_of_vehicle[f - 1]
                    false_ind = list(np.where(temp_check == False)[0])

    print('vehicle 2', vehicle_number_approximation)
    # 08/03/2020:
    # TODO: have bugs, cvrptw.visit may contains invalid x, aka (k, f)
    # 20:  vehicle{1: 5, 2: 7, 3: 8, 4: 7, 5: 6, 6: 7, 7: 7, 8: 7, 9: 6, 10: 6}
    # 50:  vehicle{1: 5, 2: 8, 3: 9, 4: 8, 5: 9, 6: 9, 7: 10, 8: 10, 9: 10, 10: 6}
    # 100: vehicle{1: 6, 2: 8, 3: 13, 4: 9, 5: 16, 6: 11, 7: 16, 8: 20, 9: 22, 10: 6}
    # 200: vehicle{1: 6, 2: 9, 3: 22, 4: 10, 5: 29, 6: 10, 7: 23, 8: 38, 9: 40, 10: 6}
    # vehicle_tmp = vehicle_number_approximation
    # vehicle_number_approximation = {1: 15, 2: 15, 3: 15, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10}
    # # vehicle_number_approximation = {1: 10, 2: 10, 3: 15, 4: 10, 5: 20, 6: 15, 7: 20, 8: 25, 9: 25, 10: 10}
    # for f in fleets:
    #     for k in set(range(1, vehicle_number_approximation[f]+1)) - set(range(1, vehicle_tmp[f]+1)):
    #         route[f, k] = []
    # print('vehicle 3', vehicle_number_approximation)

    vehicles = {j: [x for x in range(1, 1 + vehicle_number_approximation[j])] for j in fleets}
    Q = {i: np.repeat(Q_of_vehicle[i - 1], vehicle_number_approximation[i]) for i in fleets}
    indices = {(i, j, k, f) for f in fleets for i in nodes_F[f] for j in nodes_F[f] for k in vehicles[f] if i != j}

    initial_x = {(i, j, k, f): 0 for i, j, k, f in indices}

    print(' ')
    for key in route.keys():
        route_temp = route[key]
        if route_temp:
            for i in range(len(route_temp) - 1):
                initial_x[route_temp[i], route_temp[i + 1], key[1], key[0]] = 1
            initial_x[n + 2 * key[0] - 1, route_temp[0], key[1], key[0]] = 1
            initial_x[route_temp[len(route_temp) - 1], n + 2 * key[0], key[1], key[0]] = 1
        else:
            initial_x[n + 2 * key[0] - 1, n + 2 * key[0], key[1], key[0]] = 1  # start_depot -> end_depot

    # 200619 initial solution end

    print('initial distance:', sum((distance[i, j]) * initial_x[i, j, k, f] for i, j, k, f in indices), '\n')

    # return sum((distance[i, j]) * initial_x[i, j, k, f] for i, j, k, f in indices)

    model = cp_model.CpModel()

    x = {}
    for x_key in initial_x.keys():
        x[x_key] = model.NewBoolVar('x[%i,%i,%i,%i]' % (x_key[0], x_key[1], x_key[2], x_key[3]))
        model.AddHint(x[x_key], initial_x[x_key])
    t = {}
    for t_key in initial_t.keys():
        t[t_key] = model.NewIntVar(-3600, 10 ** 6, 't[%i,%i]' % (t_key[0], t_key[1]))
        model.AddHint(t[t_key], initial_t[t_key])

    model.Minimize(sum(distance[i, j] * x[i, j, k, f] for i, j, k, f in indices) -
                   sum(x[n + 2 * f - 1, n + 2 * f, k, f] for f in fleets for k in vehicles[f]))

    # 0.each flight will be served by a fleet one time
    for i in customers:
        for f in fleets:
            model.Add(sum(x[i, j, k, f] for j in nodes_F[f] for k in vehicles[f] if i != j) == 1)

    # 1.capacity
    for f in fleets:
        for k in vehicles[f]:
            model.Add(
                sum(D[f][i - 1] * sum(x[i, j, k, f] for j in nodes_F[f] if i != j) for i in customers) <= Q[f][k - 1])

    # 2.start from depot and return to depot
    for f in range(1, 1 + F):
        for k in vehicles[f]:
            model.Add(
                sum(x[n + 2 * f - 1, j, k, f] for j in nodes_F[f] if j != (n + 2 * f - 1)) == 1)
            model.Add(
                sum(x[i, n + f * 2, k, f] for i in nodes_F[f] if i != (n + f * 2)) == 1)

            model.Add(
                sum(x[n + 2 * f, j, k, f] for j in nodes_F[f] if j != (n + 2 * f)) == 0)
            model.Add(
                sum(x[i, n + f * 2 - 1, k, f] for i in nodes_F[f] if i != (n + f * 2 - 1)) == 0)

    # 3.in and out
    for f in range(1, 1 + F):
        for h in customers:
            for k in vehicles[f]:
                model.Add((sum(x[i, h, k, f] for i in nodes_F[f] if i != h) -
                           sum(x[h, j, k, f] for j in nodes_F[f] if j != h) == 0))

    # 4.TW start
    # TW start
    for f in range(1, 1 + F):
        # 10 ** 6 --> 10 ** 8 20191030
        for i in nodes_F[f]:
            for j in nodes_F[f]:
                if i != j:
                    for k in vehicles[f]:
                        model.Add(t[i, f] + tau[i, j, f] - (10 ** 8) * (1 - x[i, j, k, f]) <= t[j, f])

    for i in nodes_F[1]:
        model.Add(start_ea[i] <= t[i, 1])

    # 5.TW end - 10
    for i in nodes_F[10]:
        model.Add(t[i, 10] <= start_la[i]-S_time[10][i-1])

    # 6.precedence
    operation_type = yx[2][0][1:len(yx[2][0])]
    # retrieve: operation: fleet_id {'IB': 1, 'UL/L': 2, 'CI': 3, 'DB': 4, ...}
    retrieve = dict(zip(operation_type, range(1, len(operation_type) + 1)))
    from_task = yx[1][1][1:len(yx[1][1])]
    to_task = yx[1][2][1:len(yx[1][2])]
    ty = yx[1][0][1:len(yx[1][0])]
    ty = [int(x) for x in ty]

    precedence = {}
    for i, f_type in enumerate(type_in_schedule, start=1):
        index = [index for index, type_x in enumerate(ty) if type_x == f_type]
        precedence_item = []
        for j in index:
            precedence_item.append([retrieve[from_task[j]], retrieve[to_task[j]]])
        precedence[i] = precedence_item

    for i in customers:
        for precedence_item in precedence[i]:
            model.Add(t[i, precedence_item[0]] + S_time[precedence_item[0]][i - 1] <= t[i, precedence_item[1]])

    or_tools_solver = cp_model.CpSolver()
    or_tools_solver.parameters.log_search_progress = True
    or_tools_solver.parameters.num_search_workers = 32

    if scale == 20:
        or_tools_solver.parameters.max_time_in_seconds = 400
    elif scale == 50:
        or_tools_solver.parameters.max_time_in_seconds = 3000
    elif scale == 100:
        or_tools_solver.parameters.max_time_in_seconds = 5400
    elif scale == 200:
        or_tools_solver.parameters.max_time_in_seconds = 10000

    status = or_tools_solver.Solve(model)
    final_obj = or_tools_solver.ObjectiveValue()
    print(f"{schFilePath} Objective: {final_obj}")

    return final_obj


if __name__ == '__main__':
    dir = "./data1/20_Test1"
    scale = 20

    prjCfgPath = "./test_200304/testPrj.cfg"
    netFilePath = "./test_200304/airport.net.xml"
    opnFilePath = "./test_200304/operation_W.xlsx"
    files = os.listdir(dir)
    res_list = []
    for file in files:
        seed_everything(seed=2020)
        if os.path.splitext(file)[-1][1:] not in ["xlsx"]:
            print("Unsupported file detected!")
            continue
        path = os.path.join(dir, file)
        res = solveProject(prjCfgPath, path, opnFilePath, netFilePath, scale)
        res_list.append(res)
    print(f">> AVG Obj over {len(res_list)} instances: {sum(res_list)/len(res_list)}")
