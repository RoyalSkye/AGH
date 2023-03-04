#!/usr/bin/env python

from alns_cvrptw import *
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def used_vehicle(cvrptw):
    vehicle_distance = {(k, f): 0 for f in cvrptw.args["fleets"] for k in cvrptw.args["vehicles"][f]}
    for key, value in cvrptw.visit.items():
        if value != 1:
            continue
        i, j, k, f = key[0], key[1], key[2], key[3]
        vehicle_distance[(k, f)] += cvrptw.args["distance"][(i, j)]
    count = 0
    for key, value in vehicle_distance.items():
        if value != 0:
            count += 1
    return count


def save_sol(x, t, filepath="./res.json"):
    sol = {}
    sol["x"] = str(x)
    sol["t"] = str(t)
    try:
        with open(filepath, 'w') as j:
            json.dump(sol, j)
    except:
        print("[!] --Save Error--")
        print(" ")
    else:
        print("Saved current solution to {}".format(filepath))


def load_sol(cvrptw, filepath="./res1.json"):
    with open(filepath, 'r') as j:
        sol = json.load(j)
    cvrptw.set_solution(eval(sol["x"]), eval(sol["t"]))
    if not cvrptw.is_valid():
        print("Loaded solution {} doesn't pass constraint test!".format(cvrptw.obj))
        # print(cvrptw.visit[20, 26, k, 3])
        print(cvrptw.time[20, 3], cvrptw.args["travel_time"][20, 26, 3], cvrptw.time[26, 3])
        exit(0)
    print("Loaded a solution from {}, Objective is {}, Used {} vehicles".format(filepath, cvrptw.obj, used_vehicle(cvrptw)))
    return cvrptw


class MyCallback(NodeCallback):
    def __init__(self, env):
        NodeCallback.__init__(self, env)
        print("TEST CALLBACK")

    def __call__(self):
        print("before___________call_____________")
        print(self.get_branch_variable(0))
        print(self.get_branch_variable(1))
        print("after___________call_____________")
        # import cplex
        # cplex.advanced.strong_branching()
        return


# TODO: Remove
def old_random_destroy(cvrptw, degree_of_destruction=0.1, mipgap=0.1, timelimit=300):
    x_len = len(cvrptw.visit)
    x_fixed = random.sample(list(cvrptw.visit), int((1 - degree_of_destruction) * x_len))

    # x = {}
    # a = time.time()
    # for i, x_key in enumerate(pre_x.keys()):
    #     # name = "x" + str(i)
    #     if x_key not in x_fixed:  # fix part of variable by setting lb and ub
    #         x_val = pre_x[x_key]
    #         x_item = mdl.integer_var(lb=x_val, ub=x_val)
    #     else:
    #         x_item = mdl.integer_var(lb=0, ub=1)
    #     x[x_key] = x_item
    # for i, (k, v) in enumerate(x.items()):
    #     print(k, v)
    #     print(type(v))
    # print(time.time() - a)

    # print(len(cvrptw.visit) + len(cvrptw.time))
    # processes = []
    # for i in range(10):
    #     p = multiprocessing.Process(target=Multi_construct_cplex_model, args=(cvrptw, mdl, x, t, x_fixed, i+1))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    # Solve within timelimit
    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    # mdl.parameters.mip.strategy.variableselect = 3
    # disable primal heuristics
    # mdl.parameters.mip.strategy.heuristiceffort = 0
    # or mdl.parameters.mip.strategy.heuristicfreq = -1
    # mdl.parameters.mip.limits.strongcand = 10
    # mdl.register_callback(MyCallback)
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()


def test1(cvrptw, degree_of_destruction=0.1, mipgap=0.1, timelimit=300):
    """
    Construct model from scratch each time.
    """
    x_len = len(cvrptw.visit)
    x_fixed = random.sample(list(cvrptw.visit), int((1 - degree_of_destruction) * x_len))
    pre_x = cvrptw.visit
    mdl = Model('CVRP_multi_fleet')
    # mdl = myModel('CVRP_multi_fleet')
    a1 = time.time()
    test1 = time.time()
    x = mdl.binary_var_dict(cvrptw.visit, name='x')
    cvrptw.x = x
    print("Construct x within {}s".format(time.time() - test1))
    # solve t from scratch
    test1 = time.time()
    t = mdl.integer_var_dict(cvrptw.time, name='t')
    cvrptw.t = t
    # t = mdl.continuous_var_dict(cvrptw.time, name='t')
    print("Construct t within {}s".format(time.time() - test1))
    test1 = time.time()
    mdl = add_constraints(mdl, x, t, cvrptw)
    print("Construct constraints within {}s".format(time.time() - test1))
    print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))
    test1 = time.time()
    lns_constraints = mdl.add_constraints([x[key] == pre_x[key] for key in x_fixed])  # fix part of variable by add_constraints
    print("Add LNS Constraints within {}s".format(time.time() - test1))
    print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    # print(type(lns_constraints), len(lns_constraints))
    # test1 = time.time()
    # for i in range(len(x_fixed)):  # too slow
    #     mdl.remove_constraint(str(i))
    # mdl.remove_constraints(x[key] == pre_x[key] for key in x_fixed)  # Error
    # mdl.remove_constraints(lns_constraints)
    # print("Remove LNS Constraints within {}s".format(time.time() - test1))
    # print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))

    test1 = time.time()
    n = len(cvrptw.args["customers"])
    mdl.minimize(mdl.sum((cvrptw.args["distance"][i, j]) * x[i, j, k, f] for i, j, k, f in cvrptw.visit)
                 - mdl.sum(x[n + 2 * f - 1, n + 2 * f, k, f] for f in cvrptw.args["fleets"] for k in cvrptw.args["vehicles"][f]))
    print("Construct Obj within {}s".format(time.time() - test1))
    print("Construct Cplex model Within {}s".format(time.time() - a1))
    cvrptw.construct_time = cvrptw.construct_time + time.time() - a1

    # Solve within timelimit
    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    # mdl.parameters.mip.strategy.variableselect = 3
    # disable primal heuristics
    # mdl.parameters.mip.strategy.heuristiceffort = 0
    # or mdl.parameters.mip.strategy.heuristicfreq = -1
    # mdl.parameters.mip.limits.strongcand = 10
    # mdl.register_callback(MyCallback)
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()

    return solution


def test2(cvrptw, degree_of_destruction=0.1, mipgap=0.1, timelimit=300):
    """
    Write/Load Model from File.
    """
    # TODO: a disjoint union & Fix which part of variables? e.g. i=1
    x_len = len(cvrptw.visit)
    x_fixed = random.sample(list(cvrptw.visit), int((1 - degree_of_destruction) * x_len))
    pre_x = cvrptw.visit

    # 06/10/2020: Construct model
    a1 = time.time()
    if tmp == 0:
        mdl = Model('CVRP_multi_fleet')
        b1 = time.time()
        x = mdl.binary_var_dict(cvrptw.visit, name='x')
        print("len(x) = {}".format(len(x)))
        print("Construct x within {}s".format(time.time() - b1))
        # solve t from scratch
        b2 = time.time()
        t = mdl.integer_var_dict(cvrptw.time, name='t')
        print("len(t) = {}".format(len(t)))
        # t = mdl.continuous_var_dict(cvrptw.time, name='t')
        print("Construct t within {}s".format(time.time() - b2))
        b3 = time.time()
        mdl = add_constraints(mdl, x, t, cvrptw)
        print("Construct constraints within {}s".format(time.time() - b3))
        test1 = time.time()
        mdl.export_as_lp("./base_cplex_model.lp")
        print("Write base model to file within {}s".format(time.time() - test1))
    else:
        test2 = time.time()
        mr = ModelReader()
        mdl = mr.read_model("./base_cplex_model.lp", model_name="CVRP_multi_fleet", verbose=True)
        print("Load base model within {}s".format(time.time() - test2))
    # END 06/11/2020

    # Note: cannot reuse binary_var_dict-x, since it belongs to the previous mdl.
    test3 = time.time()
    mdl.add_constraints(mdl.get_var_by_name("x_{}_{}_{}_{}".format(key[0], key[1], key[2], key[3])) == pre_x[key] for key in x_fixed)
    print("Add LNS Constraints within {}s".format(time.time() - test3))
    # mdl.add_constraints(x[key] == pre_x[key] for key in x_fixed)  # fix part of variable by add_constraints
    print("constraints: {}, variables: {}".format(mdl.number_of_constraints, mdl.number_of_variables))
    n = len(cvrptw.args["customers"])
    test4 = time.time()
    mdl.minimize(mdl.sum((cvrptw.args["distance"][i, j]) * mdl.get_var_by_name("x_{}_{}_{}_{}".format(i, j, k, f)) for i, j, k, f in cvrptw.visit)
                 - mdl.sum(mdl.get_var_by_name("x_{}_{}_{}_{}".format(n + 2 * f - 1, n + 2 * f, k, f)) for f in cvrptw.args["fleets"] for k in cvrptw.args["vehicles"][f]))
    print("Construct Obj within {}s".format(time.time() - test4))
    # mdl.minimize(mdl.sum((cvrptw.args["distance"][i, j]) * x[i, j, k, f] for i, j, k, f in cvrptw.visit)
    #              - mdl.sum(x[n + 2 * f - 1, n + 2 * f, k, f] for f in cvrptw.args["fleets"] for k in cvrptw.args["vehicles"][f]))
    cvrptw.construct_time = cvrptw.construct_time + time.time() - a1
    print("Construct Cplex model Within {}s".format(time.time() - a1))

    # Solve within timelimit
    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()

    # Collect solution value
    test5 = time.time()
    x_sol, t_sol = {}, {}
    for i in cvrptw.visit:
        x_sol[i] = mdl.get_var_by_name("x_{}_{}_{}_{}".format(i[0], i[1], i[2], i[3])).solution_value
    for i in cvrptw.time:
        t_sol[i] = mdl.get_var_by_name("t_{}_{}".format(i[0], i[1])).solution_value
    print("Collect Solution within {}s".format(time.time() - test5))

    return solution, x_sol, t_sol


def by_customer_destroy(cvrptw, degree_of_destruction=0.1, mipgap=0.1, timelimit=300, tmp=0):
    """
    Decompostion by Customer.
    """
    fix_num = int(len(cvrptw.args["customers"]) * (1-degree_of_destruction))
    fix_id = random.sample(cvrptw.args["customers"], fix_num)
    x_fixed = [(i, j, k, f) for f in cvrptw.args["fleets"] for i in cvrptw.args["nodes_fleet"][f] for j in cvrptw.args["nodes_fleet"][f] for k in cvrptw.args["vehicles"][f] if i != j and (i in fix_id and j in fix_id)]
    print("rate of fixing: {}/{} = {}".format(len(x_fixed), len(cvrptw.visit), len(x_fixed)/len(cvrptw.visit)))
    # 1 = x_fixed + x_destroy + depot-depot
    destroy_id = list(set(cvrptw.args["customers"]) - set(fix_id))
    x_destroy = [(i, j, k, f) for f in cvrptw.args["fleets"] for i in cvrptw.args["nodes_fleet"][f] for j in cvrptw.args["nodes_fleet"][f] for k in cvrptw.args["vehicles"][f] if i != j and (i in destroy_id or j in destroy_id)]
    print("rate of destroying: {}/{} = {}".format(len(x_destroy), len(cvrptw.visit), len(x_destroy)/len(cvrptw.visit)))

    # Construct model
    mdl = construct_cplex_model(cvrptw, x_fixed)

    mdl.parameters.timelimit = timelimit
    mdl.parameters.mip.tolerances.mipgap = mipgap
    mdl.parameters.advance = 0  # Do not use advanced start information, start the new solve from scratch
    solution = mdl.solve(log_output=True)
    mdl.get_solve_status()

    return solution
# TODO END


def create_datasets(data, unshared_data, file_map, output_folder="./data", file_count=0, shared_data=True):
    """
    two dataset files:
    1. ID_MAP
    2. CVRPTW
    """
    # convert key(tuple) -> key(str)
    t = time.time()
    print("Generating Dataset...")
    if shared_data:
        with open(os.path.join(output_folder, f"SHARED_DATA_{file_count}.json"), 'w') as f:
            for _, value in data.items():
                for k, v in value.items():
                    if k == "id_map" or k == "veh_map":
                        value[k] = {str(key): v[key] for key in v}
            json.dump(data, f)
    with open(os.path.join(output_folder, f"UNSHARED_DATA_{file_count}.json"), 'w') as f:
        json.dump(unshared_data, f)
    with open(os.path.join(output_folder, f"FILE_MAP_{file_count}.json"), 'w') as f:
        json.dump(file_map, f)
    print("Dataset -> {} within {}s".format(output_folder, int(time.time() - t)))


def load_datasets(dir_path="./data", file_count=0):
    file_path = os.path.join(dir_path, f"FILE_MAP_{file_count}.json")
    shared_data_path = os.path.join(dir_path, f"SHARED_DATA_{file_count}.json")
    unshared_data_path = os.path.join(dir_path, f"UNSHARED_DATA_{file_count}.json")
    with open(file_path, "r") as f:
        file_map = json.load(f)
    with open(shared_data_path, "r") as f:
        shared_data = json.load(f)
        # print(shared_data["0"].keys())
        # shared_data = {str2tuple(k): veh_map[k] for k in veh_map}
    with open(unshared_data_path, "r") as f:
        unshared_data = json.load(f)

    return file_map, unshared_data, shared_data


def str2tuple(str):
    # (i, j, k, f) / (i, f) / (k, f)
    str = str[1: -1].replace(" ", "")
    a = str.split(",")
    return tuple([int(i) for i in a])


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(data_name, epoch, model, optimizer, best_loss, best_acc, is_best, epochs_since_improvement):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer to update model's weights
    :param is_best: is this checkpoint the best so far?
    """
    dir = './models'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    filename = "checkpoint_{}_{}.pth.tar".format(data_name, epoch)
    best_filename = "BEST_checkpoint_{}.pth.tar".format(data_name)
    filename = os.path.join(dir, filename)
    best_filename = os.path.join(dir, best_filename)

    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'best_loss': best_loss,
             'best_acc': best_acc,
             "epochs_since_improvement": epochs_since_improvement}

    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, best_filename)

    return filename


def show(x, y, label, title, xdes, ydes, path, x_scale="linear", dpi=600):
    # plt.style.use('ggplot')
    plt.figure(figsize=(10, 8))
    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:cyan',
              'tab:gray', 'tab:brown', 'tab:purple', 'tab:olive', 'tab:pink']
    # colors = ['tab:green', 'tab:orange', 'tab:green', 'tab:orange', 'tab:green',
    #           'tab:orange', 'tab:brown', 'tab:purple', 'tab:olive', 'tab:pink']
    # plt.xticks(rotation=45)
    plt.rcParams.update({'font.size': 20})

    assert len(x) == len(y)
    for i in range(len(x)):
        if i < len(label):
            plt.plot(x[i], y[i], color=colors[i], label=label[i])
        else:
            plt.plot(x[i], y[i], color=colors[i % len(label)])
        # marker="o", linestyle='dashed'


    # plt.xlim(0, 30000000)
    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes)
    plt.ylabel(ydes)

    # plt.title(title)
    plt.legend(loc='upper right', fontsize=16)  # 20 for cplex_vs_LNS_XXX
    plt.xscale(x_scale)

    # plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close("all")


def pre_pre_process(x, scale):
    X = []
    for i in x:
        x1 = []
        start, end = 0, 1
        for ii in i:
            if end >= len(scale):
                print(f">> OUT OF RANGE! end is {end}")
                break
            if ii <= scale[start]:
                x1.append((ii-scale[start])/(scale[end]-scale[end])+start)
            else:
                start += 1
                end += 1
        X.append(x1)

    return X


def pre_process(x, y, threshold=10**6):
    # pre-process data for 'show' func.
    assert len(x) == len(y)
    X, Y = [], []
    for i, j in zip(x, y):
        incumbent = 10 ** 8
        tmp_x, tmp_y = [], []
        for c in range(len(i)):
            if i[c] > threshold:
                break
            incumbent = min(incumbent, j[c])
            tmp_x.append(i[c])
            tmp_y.append(incumbent)
            if c + 1 < len(i):
                if i[c+1] > threshold:
                    tmp_x.append(threshold)
                else:
                    tmp_x.append(i[c+1])
                tmp_y.append(incumbent)
        X.append(tmp_x)
        Y.append(tmp_y)
    return X, Y


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("[*] DECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))


def seed_everything(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='LNS')
    # Data parameters
    parser.add_argument('--data_folder', default="./data/20_Test", help='folder with data files')
    parser.add_argument('--prjCfgPath', default="./test_200304/testPrj.cfg", help='folder with data files')
    parser.add_argument('--opnFilePath', default="./test_200304/operation_W.xlsx", help='folder with data files')
    parser.add_argument('--netFilePath', default="./test_200304/airport.net.xml", help='folder with data files')
    parser.add_argument('--resultFilePath', default="./test_200304/result.xlsx", help='folder with data files')
    parser.add_argument('--checkpoint', default="./models/checkpoint_FT_6.pth.tar", help='path to checkpoint, None if none.')
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

    return args


def compute_incumbents_avg(vehicle_random_1, forward_training_1):
    """
        Pre-process to incumbents, then avg
    """
    # pre-process obj to incumbents
    vehicle_random, forward_training = [], []
    for i in vehicle_random_1:
        incumbent = 10 ** 7
        ll = []
        for j in i:
            j = 10 ** 8 if j is None else j
            incumbent = min(j, incumbent)
            ll.append(incumbent)
        vehicle_random.append(ll)
    for i in forward_training_1:
        incumbent = 10 ** 7
        ll = []
        for j in i:
            j = 10 ** 8 if j is None else j
            incumbent = min(j, incumbent)
            ll.append(incumbent)
        forward_training.append(ll)

    vehicle_random_avg, forward_training_avg = [], []
    for i in range(len(vehicle_random[0])):
        sum1, sum2 = 0, 0
        for j in range(len(vehicle_random)):
            sum1 += vehicle_random[j][i]
            sum2 += forward_training[j][i]
        vehicle_random_avg.append(sum1 / len(vehicle_random))
        forward_training_avg.append(sum2 / len(vehicle_random))

    print(vehicle_random_avg)
    print(forward_training_avg)

    return [vehicle_random_avg, forward_training_avg]
