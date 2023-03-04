from docplex.cp.model import CpoModel
# input: 
# customers list
# vehicle list
# visit_fixed =  {(cust1, cust2, vehicle): visit/no visit}
# time_fixed = {(cust, vehicle): time}
# vehicle capacity (int) speed (int)
# cust resources  (dict)
# service time (dict)
# time window (dict of tuple)
# arr_dep (arrival, departure)
def LNS_VRPTW(visit_fixed, time_fixed, distance, custs, vehicles, capacity, speed, resources, service_time, time_win):
    custs_w_depot = [0] + custs + [len(custs) + 1]
    time_min = min(time_win.values(), key = lambda x: x[0])[0]
    time_max = max(time_win.values(), key = lambda x: x[1])[1]
    
    mdl = CpoModel("VRPTW")
    # decision variables
    visit = [(i,j,k) for i in custs_w_depot for j in custs_w_depot for k in vehicles]
    time = [(i, k) for i in custs for k in vehicles]
    visit_var = mdl.integer_var_dict(visit, name = "visit")
    time_var = mdl.integer_var_dict(time, name = "time")

    # add constraints
    # Range
    for key in visit_var:
        mdl.add(visit_var[key]>=0)
        mdl.add(visit_var[key]<=1)
    # route constraints
    for i in custs:
        mdl.add(mdl.sum(visit_var[(i, j, k)] for j in custs_w_depot for k in vehicles) == 1)
    # capacity
    for k in vehicles:
        mdl.add(mdl.sum(visit_var[(i, j, k)] * resources[i] for i in custs for j in custs_w_depot) <= capacity)
    # from depot
    for k in vehicles:
        mdl.add(mdl.sum(visit_var[(0, j, k)] for j in custs) == 1)
        mdl.add(mdl.sum(visit_var[(i, 0, k)] for i in custs) == 0)
    # flow
    for k in vehicles:
        for p in custs:
            mdl.add(mdl.sum(visit_var[(i, p, k)] for i in custs_w_depot) == mdl.sum(visit_var[p, j, k] for j in custs_w_depot))
    # to depot
    for k in vehicles:
       mdl.add(mdl.sum(visit_var[(i, len(custs)+1, k)] for i in custs) == 1)
       mdl.add(mdl.sum(visit_var[(len(custs)+1, j, k)] for j in custs) == 0)
    # cannot start and end from self
    for k in vehicles:
        for i in custs:
            mdl.add(visit_var[i, i, k] == 0)
    # time constraint between 2 customers
    for i in custs:
        for j in custs:
            for k in vehicles:
                mdl.add(visit_var[(i, j, k)]*(time_var[i, k] + distance[i][j]/speed - time_var[j, k]) <= 0)
    # time window constraint
    for i in custs:
        mdl.add(mdl.sum(visit_var[i, j, k] * time_var[i, k] for j in custs_w_depot for k in vehicles) >= time_win[i][0])
        mdl.add(mdl.sum(visit_var[i, j, k] * time_var[i, k] for j in custs_w_depot for k in vehicles) <= time_win[i][1])
    # time within range
    for i in custs:
        mdl.add(mdl.sum(visit_var[i, j, k] * time_var[i, k] for j in custs_w_depot for k in vehicles) >= time_min)
        mdl.add(mdl.sum(visit_var[i, j, k] * time_var[i, k] for j in custs_w_depot for k in vehicles) <= time_max)
        

    # known route
    for key in visit_fixed:
        mdl.add(visit_var[key] == visit_fixed[key])
    for key in time_fixed:
        mdl.add(time_var[key] == time_fixed[key])
        

    # add objective
    obj = mdl.minimize(mdl.sum(visit_var[(i, j, k)] * distance[i][j] for i in custs_w_depot for j in custs_w_depot for k in vehicles))
    mdl.add(obj)

    # time_lim_set = {20: 5, 50: 10, 100: 60, 200: 180}
    # time_lim = time_lim_set[len(custs)]
    time_lim = 30

    
    sol = mdl.solve(TimeLimit = time_lim, LogVerbosity = "Quiet", agent='local',
               execfile='/home/hw1-a30/CPLEX_Studio1210/cpoptimizer/bin/x86-64_linux/cpoptimizer')
    # sol = mdl.solve(TimeLimit = 5, LogVerbosity = "Verbose", agent='local',
    #            execfile='/Applications/CPLEX_Studio201/cpoptimizer/bin/x86-64_osx/cpoptimizer')
    if not sol:
        return None
    # sol.write()

    # new route info
    # distance
    distance_result = sol.get_objective_values()[0]
    if (type(distance_result) == tuple):
        distance_result = distance_result[0]
    # print("distance:", distance_result)
    # route
    routes = []
    for k in vehicles:
        # route_seq = [(i, j) for i in custs_w_depot for j in custs_w_depot \
        #     if sol.get_var_solution(visit_var[i, j, k]).get_value() != 0]
        # print("route", route_seq)
        time = [(i, sol.get_var_solution(time_var[i, k]).get_value()) for i in custs \
            if time_min <= sol.get_var_solution(time_var[i, k]).get_value() <= time_max]
        time.sort(key = lambda x: x[1])
        route = [{"id": x[0], "begin_time": x[1]} for x in time]
        routes.append(route)
    return routes, distance_result
    
