from docplex.mp.model import Model
# input: 
# customers list
# vehicle list
# visit_fixed =  {(cust1, cust2, vehicle): visit/no visit}
# time_fixed = {(cust, vehicle): time}
# vehicle capacity (int) speed (int)
# cust resources  (dict)
# service time (dict)
# time window (dict of tuple)
# distance
def LNS_VRPTW(visit_fixed, time_fixed, distance, custs, vehicles, capacity, speed, resources, service_time, time_win):
    custs_w_depot = [0] + custs + [len(custs) + 1]
    mdl = Model("VRPTW")
    # decision variables
    visit = [(i,j,k) for i in custs_w_depot for j in custs_w_depot for k in vehicles]
    time = [(i, k) for i in custs for k in vehicles]
    visit_var = mdl.binary_var_dict(visit, name = "visit")
    time_var = mdl.binary_var_dict(time, name = "time")

    # add constraints
    # route constraints
    mdl.add_constraints(mdl.sum(visit_var[(i, j, k)] for j in custs_w_depot for k in vehicles) == 1 for i in custs)
    mdl.add_constraints(mdl.sum(visit_var[(i, j, k)] * resources[i] for i in custs for j in custs_w_depot) <= capacity for k in vehicles)
    mdl.add_constraints(mdl.sum(visit_var[(0, j, k)] for j in custs) == 1 for k in vehicles)
    mdl.add_constraints(mdl.sum(visit_var[(i, p, k)] for i in custs_w_depot) == mdl.sum(visit_var[p, j, k] for j in custs_w_depot) for p in custs for k in vehicles)
    mdl.add_constraints(mdl.sum(visit_var[(i, len(custs)+1, k)] for i in custs) == 1 for k in vehicles)
    for i in custs:
        for j in custs:
            for k in vehicles:
                mdl.add_constraint(visit_var[(i, j, k)]*(time_var[i, k] + distance[i][j]/speed - time_var[j, k]) <= 0)
    # mdl.add_constraints(visit_var[(i, j, k)]*(time_var[i, k] + distance[i][j]/speed - time_var[j, k]) <= 0 for i in custs for j in custs for k in vehicles)
    mdl.add_constraints(time_var[(i, k)] >= time_win[i][0] for i in custs for k in vehicles)
    mdl.add_constraints(time_var[(i, k)] <= time_win[i][1] for i in custs for k in vehicles)
    # known route
    mdl.add_constraints(visit_var[key] == visit_fixed[key] for key in visit_fixed)
    mdl.add_constraints(time_var[key] == time_fixed[key] for key in time_fixed)

    # add objective
    mdl.minimize(mdl.sum(visit_var[(i, j, k)] * distance[i][j] for i in custs_w_depot for j in custs_w_depot for k in vehicles))
    solution = mdl.solve()
    solution.print_information()
    
