from docplex.cp.model import CpoModel

# Time window shceduling
# Given: earliest and latest time that all tasks need to be completed within
# Constraints: precedence, excat time of performance of some tasks
# Return: the largest possible time window

# Task = namedtuple("Task", "duration precedence_before precedence_after exact_time")

# def tw_schedule(time_slot, tasks):
#
#     # Find earliest start time:
#     mdl, task_vars = load_model(time_slot, tasks)
#     # Add objective:
#     obj = mdl.minimize(sum([mdl.start_of(task_vars[id]) for id in tasks]))
#     mdl.add(obj)
#
#     # Solve model
#     # sol = mdl.solve(TimeLimit = 10, LogVerbosity = "Verbose")
#     sol = mdl.solve(TimeLimit = 10, LogVerbosity = "Quiet")
#
#     if sol:
#         # sol.write()
#         time_window = {id: [sol.get_var_solution(task_vars[id]).get_start()] for id in tasks}
#     else:
#         print("Instance not feasible - earliest start time")
#         return {}
#
#     # Find Latest start time:
#     mdl, task_vars = load_model(time_slot, tasks)
#     # Add objective:
#     obj = mdl.maximize(sum([mdl.start_of(task_vars[id]) for id in tasks]))
#     mdl.add(obj)
#
#     # Solve model
#     # sol = mdl.solve(TimeLimit = 10, LogVerbosity = "Verbose")
#     sol = mdl.solve(TimeLimit = 10, LogVerbosity = "Quiet")
#
#     if sol:
#         # sol.write()
#         for id in tasks:
#             time_window[id].append(sol.get_var_solution(task_vars[id]).get_start())
#     else:
#         print("Instance not feasible - latest start time")
#         return {}
#     print(time_window)
#     return time_window
#
# def load_model(time_slot, tasks):
#     start_time = time_slot[0]
#     end_time = time_slot[1]
#
#     mdl = CpoModel(name = "generate_tw")
#     # Create model for CP and solve for the first time to obtain initial time windows
#     # Decision variables: time interval of each task
#     task_vars = {}
#     for id in tasks:
#         if tasks[id].exact_time != None:
#             print("exact time: {}".format(tasks[id].exact_time))
#             task_vars[id] = mdl.interval_var(start = tasks[id].exact_time, size = tasks[id].duration, name = "task_"+str(id))
#         else:
#             task_vars[id] = mdl.interval_var(size = tasks[id].duration, name = "task_"+ str(id))
#     slot = mdl.interval_var(start = start_time , end = end_time) # Time interval of entire time slot
#     # Add Constraints:
#     for id in tasks:
#         # 1st constraint: all intervals are within start and end time
#         mdl.add(mdl.start_before_start(slot, task_vars[id]))
#         mdl.add(mdl.end_before_end(task_vars[id], slot))
#         # mdl.add(mdl.span(slot, task_vars.values()))
#
#         if tasks[id].precedence_before != []:
#             # Tasks that have precedence_before constraints
#             # 2nd Constraints: end of precedence_before_task is before start of current task
#             for id_before in tasks[id].precedence_before:
#                 mdl.add(mdl.end_before_start(task_vars[id_before], task_vars[id]))
#     return mdl, task_vars

def tw_schedule(time_slot, tasks):
    start_time = time_slot[0]
    end_time = time_slot[1]

    mdl = CpoModel(name = "generate_tw")
    # Create model for CP and solve for the first time to obtain initial time windows
    # Decision variables: time interval of each task
    task_vars = {}
    early = lambda x: "task_" + str(x) + "_early"
    late = lambda x: "task_" + str(x) + "_late"
    for id in tasks:
        if tasks[id].exact_time != None:
            # print("exact time: {}".format(tasks[id].exact_time))
            task_vars[early(id)] = mdl.interval_var(start = tasks[id].exact_time, size = tasks[id].duration, name = early(id))
            task_vars[late(id)] = mdl.interval_var(start = tasks[id].exact_time, size = tasks[id].duration, name = late(id))
        else:
            task_vars[early(id)] = mdl.interval_var(size = tasks[id].duration, name = early(id))
            task_vars[late(id)] = mdl.interval_var(size = tasks[id].duration, name = late(id))
    slot = mdl.interval_var(start = start_time , end = end_time) # Time interval of entire time slot
    # Add Constraints:
    for id in tasks:
        # 1st constraint: all intervals are within start and end time
        mdl.add(mdl.start_before_start(slot, task_vars[early(id)]))
        mdl.add(mdl.end_before_end(task_vars[early(id)], slot))
        mdl.add(mdl.start_before_start(slot, task_vars[late(id)]))
        mdl.add(mdl.end_before_end(task_vars[late(id)], slot))
        # mdl.add(mdl.span(slot, task_vars.values()))

        if tasks[id].precedence_before != []:
            # Tasks that have precedence_before constraints
            # 2nd Constraints: end of precedence_before_task is before start of current task
            for id_before in tasks[id].precedence_before:
                mdl.add(mdl.end_before_start(task_vars[early(id_before)], task_vars[early(id)]))
                mdl.add(mdl.end_before_start(task_vars[late(id_before)], task_vars[late(id)]))

    # Add objective:
    obj = mdl.maximize(sum([mdl.start_of(task_vars[late(id)]) - mdl.start_of(task_vars[early(id)]) for id in tasks]))
    mdl.add(obj)

    # Solve model
    # sol = mdl.solve(TimeLimit = 10, LogVerbosity = "Verbose")
    sol = mdl.solve(TimeLimit = 10, LogVerbosity = "Quiet", agent='local',
               execfile='/home/hw1-a30/CPLEX_Studio1210/cpoptimizer/bin/x86-64_linux/cpoptimizer')
    # sol = mdl.solve()

    if sol:
        # sol.write()
        time_window = {id: (sol.get_var_solution(task_vars[early(id)]).get_start(),
                sol.get_var_solution(task_vars[late(id)]).get_start()) for id in tasks}
    else:
        print("Instance not feasible - earliest start time")
        return {}

    return time_window
