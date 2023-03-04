# customers: {customer id: time_window duration resource} time_window - (left, right)
# fleet: name num capacity speed
# distance between i and j (depot - 0): distance[i][j]
import copy
import random
import datetime
import matplotlib.pyplot as plt
from lns import LNS_VRPTW

class VRPTW:
    def __init__(self, customers = None, fleet = None, distance = None):
        self.customers = customers
        self.fleet = fleet
        self.distance = distance
        self.routes = []
        self.cust_begin_time = {}
        self.results = {}

    def load_customers(self, customers):
        self.customers = customers

    def load_fleet(self, fleet):
        self.fleet = fleet

    def load_distance(self, distance):
        self.distance = distance

    def routing(self):
        all_id = self.customers.keys()
        remaining_id = set(all_id)
        served_id = []
        while len(remaining_id) != 0:
            remaining_custs = {id: self.customers[id] for id in remaining_id}
            # New route
            route = []
            # Choose an initial customer to be inserted - having the earliest start time
            init_customer_id = min(remaining_custs, key = lambda id: self.customers[id].time_window[0])
            init_customer_info = self.customers[init_customer_id]
            init_customer = {
                    "id": init_customer_id,
                    "tw_earliest": init_customer_info.time_window[0],
                    "tw_latest": init_customer_info.time_window[1],
                    "duration": init_customer_info.duration,
                    "resource": init_customer_info.resource,
                    "vehicle_ready_time": 0,
                    "begin_time": init_customer_info.time_window[0],
                    "push_forward": 0
                    }
            route.append(init_customer)
            served_id.append(init_customer_id)

            while 1:
                feasible_routes = [] # tuple: (inserted customer, resultant route)
                # All possibilities to insert a customer in a position
                for id in all_id:
                    if id not in served_id:
                        # Create a customer
                        customer_info = self.customers[id]
                        customer = {
                                "id": id,
                                "tw_earliest": customer_info.time_window[0],
                                "tw_latest": customer_info.time_window[1],
                                "duration": customer_info.duration,
                                "resource": customer_info.resource,
                                "vehicle_ready_time": None,
                                "begin_time": None,
                                "push_forward": 0
                                }
                        for insert_pos in range(len(route) + 1):
                            feasible = True
                            route_candidate = self.__insert_cust(route, insert_pos, customer)
                            # Update times of the subsequent customers after the inserted customer (included)
                            for i in range(insert_pos, len(route_candidate)):
                                self.__update_time(route_candidate, i)
                            #     # Check feasibility: begin_time < latest_start_time
                            #     if route_candidate[i]["begin_time"] > route_candidate[i]["tw_latest"]:
                            #         feasible = False
                            #         break
                            #     # If push forward becomes 0, all subsequent customers are feasible
                            #     if i != insert_pos and route_candidate[i]["push_forward"] == 0:
                            #         feasible = True
                            #         break
                            # if feasible:
                            #     feasible_routes.append((insert_pos, copy.deepcopy(route_candidate)))

                            if self.__route_feasible(route_candidate):
                                feasible_routes.append((insert_pos, copy.deepcopy(route_candidate)))

                if feasible_routes != []:
                    # From feasible routes, choose the one with minimum cost function
                    chosen = min(feasible_routes, key = lambda x: self.__cost_function(x))
                    insert_pos = chosen[0]
                    route = copy.deepcopy(chosen[1])
                    served_id.append(route[insert_pos]["id"])
                else:
                    # No feasible customers can be inserted
                    break

            # End of the route
            self.__check_feasible(route)

            self.routes.append(route)
            for cust in route:
                added_cust_id = cust["id"]
                added_cust_begin_time = cust["begin_time"]
                self.cust_begin_time[added_cust_id] = int(round(added_cust_begin_time))

            remaining_id = set(all_id).symmetric_difference(set(served_id))
        # End of all routes
        print("Initial solution found.")
        self.__present_results()
        print("Performing large neighborhood search...")
        self.__lns()
        print("Improved solution found.")
        self.__present_results()

    def __lns(self):
        # Use VND: RPOP until cannot improve, then change to SMART until cannot improve

        start_time = datetime.datetime.now()
        cust_num = len(self.customers)
        # time_lim = {20: 6, 50: 30, 100: 180, 200: 900}
        # max_time = time_lim[cust_num] # LNS does not exceed 250 second
        max_time = 250
        best_route = copy.deepcopy(self.routes)
        best_distance = self.results["total_distance"]

        # print("first")
        # best_route = self.__SMART(copy.deepcopy(best_route))
        # print("second")
        # best_route = self.__RPOP(copy.deepcopy(best_route))

        run_RPOP = True
        i = 1

        while((datetime.datetime.now() - start_time).total_seconds() <= max_time):
        # for i in range(10):
            # print("Performing LNS iteration", i)
            # print("Time", (datetime.datetime.now() - start_time).total_seconds())
            i += 1
            if run_RPOP:
                # print("RPOP")
                # run RPOP
                RPOP_result = self.__RPOP(copy.deepcopy(best_route))
                if not RPOP_result:
                    # if no solution, change to another operator
                    run_RPOP = False
                else:
                    route = RPOP_result[0]
                    distance = RPOP_result[1]
                    if not distance < best_distance:
                        # if distance not improved, change to another operator
                        run_RPOP = False
                    else:
                        # if have better solution, continue with the opertaor in next iteration
                        best_route = route
                        best_distance = distance
                        # print("RPOP improved")
            else:
                # run SMART
                # print("SMART")
                SMART_result = self.__SMART(copy.deepcopy(best_route))
                if not SMART_result:
                    # if no solution, change to another operator
                    run_RPOP = True
                else:
                    route = SMART_result[0]
                    distance = SMART_result[1]
                    if not distance < best_distance:
                        # if distance not improved, change to another operator
                        run_RPOP = True
                    else:
                        # if have better solution, continue with the opertaor in next iteration
                        best_route = route
                        best_distance = distance
                        # print("SMART improved")
        # Time up
        self.routes = copy.deepcopy(best_route)


    def __RPOP(self, routes):
        # conmplete solution before destroy
        custs = list(self.customers.keys())
        custs_w_depot = [0] + custs + [len(custs) + 1]
        resources = {key: self.customers[key].resource for key in self.customers}
        service_time = {key: self.customers[key].duration for key in self.customers}
        time_win = {key: self.customers[key].time_window for key in self.customers}
        vehicles = range(len(self.routes))
        visit_fixed = {(i, j, k): 0 for i in custs_w_depot for j in custs_w_depot for k in vehicles}
        # time_fixed = {(i, k): 0 for i in custs for k in vehicles}
        time_fixed = {}
        # fill out visit and time
        for i in range(len(routes)):
            route = routes[i]
            for j in range(len(route)): 
                cust = route[j]
                if j == 0:
                    visit_fixed[(0, cust["id"], i)] = 1
                else:
                    prev_cust = route[j-1]
                    visit_fixed[(prev_cust["id"], cust["id"], i)] = 1
                if j == len((route)) - 1:
                    visit_fixed[(cust["id"], custs_w_depot[-1], i)] = 1
                time_fixed[(cust["id"], i)] = round(cust["begin_time"])
        # # LNS to repair
        # LNS_VRPTW(visit_fixed, time_fixed, self.distance, custs, vehicles, \
        #     self.fleet.capacity, self.fleet.speed, resources, service_time, time_win)


        cust_num = len(self.customers)
        rand_pivot_num = round(cust_num * 0.08)
        similar_cust_num = round(cust_num * 0.1)
        # rand_pivot_num = 1
        # similar_cust_num = 5
        # All customers
        all_cust = []
        for route in routes:
            all_cust.extend(route)
        # destroy
        # random pivot. number: (from paper: 1/10 of customers become pivot point).
        rand_pivot = random.sample(self.customers.keys(), rand_pivot_num)
        # similar customers in time
        similar_cust_cand = []
        # rank time closeness:
        for pivot in rand_pivot:
            pivot_cust = [x for x in all_cust if x["id"] == pivot][0]
            # find closest customers: amount = similar_cust_num
            cust_wo_pivot = [cust for cust in all_cust if cust['id'] not in rand_pivot]
            cust_wo_pivot.sort(key = lambda x: x["begin_time"] - pivot_cust["begin_time"])
            similar_cust_cand.extend([(x, abs(x["begin_time"] - pivot_cust["begin_time"])) for x in cust_wo_pivot])
        # pick similar customers.  If same cust, pick next
        similar_cust_cand.sort(key = lambda x: x[1])
        similar_cust = set([cust[0]['id'] for cust in similar_cust_cand[:similar_cust_num]])
        i = 1
        while (len(similar_cust) < similar_cust_num):
            similar_cust = similar_cust.union([similar_cust_cand[similar_cust_num+i][0]["id"]])
            i+=1
        removed_cust = list(rand_pivot) + list(similar_cust)
        # print("removed:", removed_cust)

        # remove chosen cust from routes 
        visit_fixed = {(i, j, k): visit_fixed[i, j, k] for i in custs_w_depot for j in custs_w_depot for k in vehicles\
            if i not in removed_cust and j not in removed_cust}
        # time_fixed = {(i, k): time_fixed[i, k] for i in custs for k in vehicles if i not in removed_cust}
        time_fixed = {key: time_fixed[key] for key in time_fixed if key[0] not in removed_cust}
        # LNS to repair
        return LNS_VRPTW(visit_fixed, time_fixed, self.distance, custs, vehicles, \
            self.fleet.capacity, self.fleet.speed, resources, service_time, time_win)

    def __SMART(self, routes):
        # conmplete solution before destroy
        custs = list(self.customers.keys())
        custs_w_depot = [0] + custs + [len(custs) + 1]
        resources = {key: self.customers[key].resource for key in self.customers}
        service_time = {key: self.customers[key].duration for key in self.customers}
        time_win = {key: self.customers[key].time_window for key in self.customers}
        vehicles = range(len(self.routes))
        visit_fixed = {(i, j, k): 0 for i in custs_w_depot for j in custs_w_depot for k in vehicles}
        # time_fixed = {(i, k): 0 for i in custs for k in vehicles}
        time_fixed = {}
        # fill out visit and time
        for i in range(len(routes)):
            route = routes[i]
            for j in range(len(route)): 
                cust = route[j]
                if j == 0:
                    visit_fixed[(0, cust["id"], i)] = 1
                else:
                    prev_cust = route[j-1]
                    visit_fixed[(prev_cust["id"], cust["id"], i)] = 1
                if j == len((route)) - 1:
                    visit_fixed[(cust["id"], custs_w_depot[-1], i)] = 1
                time_fixed[(cust["id"], i)] = round(cust["begin_time"])

        # Config
        config = {20: 1, 50: 2, 100: 2, 200: 2}
        cust_num = len(self.customers)
        rm_before_pivot = config[cust_num]
        rm_after_pivot = config[cust_num]
        # All customers
        all_cust = []
        for route in routes:
            all_cust.extend(route)
        # destroy
        # generate random pivot
        pivot_valid = False
        while not pivot_valid:
            rand_pivot = random.choice(list(self.customers.keys()))
            # check pivot is in the middle
            for route in routes:
                for i in range(len(route)):
                    if route[i]["id"] == rand_pivot:
                        if i >= rm_before_pivot and i <= len(route)-1-rm_after_pivot:
                            # print("Valid")
                            pivot_valid = True
                            pivot = rand_pivot
                            route_with_pivot = route
                            # store removed customers
                            removed_cust = [x['id'] for x in route[i-rm_before_pivot: i+rm_after_pivot+1]]
        # find next pivot (customers from other routes)
        pivot_cust = [x for x in all_cust if x["id"] == pivot][0]
        routes_wo_pivot = [r for r in routes if r is not route_with_pivot]
        sec_pivot_cand = []
        for route in routes_wo_pivot:
            # route should be long enough
            if len(route) > rm_before_pivot + rm_after_pivot:
                sec_pivot_cand.extend(route)
        sec_pivot_cand.sort(key = lambda x: abs(x["begin_time"] - pivot_cust["begin_time"]))
        sec_pivot = [x['id'] for x in sec_pivot_cand]
        pivot_valid = False
        for pivot in sec_pivot:
            if pivot_valid:
                break
            # check pivot is in the middle
            for route in routes_wo_pivot:
                for i in range(len(route)):
                    if route[i]["id"] == pivot:
                        if i >= rm_before_pivot and i <= len(route)-1-rm_after_pivot:
                            pivot_valid = True
                            # store removed customers
                            removed_cust.extend([x['id'] for x in route[i-rm_before_pivot: i+rm_after_pivot+1]])
        # print("removed_cust", removed_cust)
        # remove chosen cust from routes 
        visit_fixed = {(i, j, k): visit_fixed[i, j, k] for i in custs_w_depot for j in custs_w_depot for k in vehicles\
            if i not in removed_cust and j not in removed_cust}
        # time_fixed = {(i, k): time_fixed[i, k] for i in custs for k in vehicles if i not in removed_cust}
        time_fixed = {key: time_fixed[key] for key in time_fixed if key[0] not in removed_cust}
        # LNS to repair
        return LNS_VRPTW(visit_fixed, time_fixed, self.distance, custs, vehicles, \
            self.fleet.capacity, self.fleet.speed, resources, service_time, time_win)

    def __insert_cust(self, route, index, customer):
        new_route = copy.deepcopy(route)
        new_route.insert(index, customer)

        return new_route

    def __update_time(self, route, i):

        if i == 0:
            route[i]["vehicle_ready_time"] = 0
        else:
            route[i]["vehicle_ready_time"] = route[i - 1]["begin_time"]\
                    + route[i-1]["duration"]\
                    + self.distance[route[i-1]["id"]][route[i]["id"]] / self.fleet.speed

        new_begin_time = max(route[i]["vehicle_ready_time"], route[i]["tw_earliest"])
        if route[i]["begin_time"] != None:
            route[i]["push_forward"] = max(0, new_begin_time - route[i]["begin_time"])
        route[i]["begin_time"] = new_begin_time

    def __cost_function(self, route_tuple):
        # Parameters for I3 from 2016 paper
        alpha_1 = 0.49
        alpha_2 = 0.49
        alpha_3 = 0.02
        miu = 1

        insert_pos = route_tuple[0]
        route = route_tuple[1]
        if insert_pos == 0:
            prev_id = 0
        else:
            prev_id = route[insert_pos - 1]['id']
        id = route[insert_pos]["id"]
        if insert_pos == len(route) - 1:
            next_id = 0
        else:
            next_id = route[insert_pos + 1]['id']

        c_1 = self.distance[prev_id][id] + self.distance[id][next_id] - miu * self.distance[prev_id][next_id]
        if insert_pos == len(route) - 1:
            c_2 = 0
        else:
            c_2 = route[insert_pos + 1]["push_forward"]

        c_3 = route[insert_pos]["tw_latest"] - route[insert_pos]["begin_time"]

        return alpha_1 * c_1 + alpha_2 * c_2 + alpha_3 * c_3

    def __route_feasible(self, route):
        resource = 0
        for cust in route:
            if cust["begin_time"] > cust["tw_latest"]:
                return False
            # Resource cannot exceed capacity of the vehicle
            resource += cust["resource"]
            if resource > self.fleet.capacity:
                return False

        return True

    def __present_results(self):
        # self.__print_routes(self.routes)
        self.results["no_of_vehicle"] = len(self.routes)
        self.results["routes_with_customers"] = self.__cal_route_w_cust()
        self.results["route_distances"], self.results["total_distance"] = self.__cal_total_distance(self.routes)
    
    def __cal_route_distance(self, route):
        route_distance = sum([self.distance[route[i]['id']][route[i+1]['id']] for i in range(len(route)-1)])
        # plus depot to 1st customer and last castomer to depot
        route_distance += self.distance[0][route[0]['id']]
        route_distance += self.distance[route[-1]['id']][0]
        return route_distance

    def __cal_total_distance(self, routes):
        route_distances = []
        for route in routes:
            route_distances.append(self.__cal_route_distance(route))
        return route_distances, sum(route_distances)

    def __cal_route_distances(self, routes):
        route_distances = []
        for route in routes:
            route_distance = sum([self.distance[route[i]['id']][route[i+1]['id']]
                    for i in range(len(route)-1)])
            # plus depot to 1st customer and last castomer to depot
            route_distance += self.distance[0][route[0]['id']]
            route_distance += self.distance[route[-1]['id']][0]
            route_distances.append(route_distance)
        return route_distances

    def __print_routes(self, routes):
        count = 1
        for route in routes:
            print("+++Route {}+++".format(count))
            count += 1
            for cust in route:
                print("Customer {}: {}".format(cust['id'], cust['begin_time']))

    def __cal_route_w_cust(self):
        routes_with_customers = []
        for route in self.routes:
            route_with_customers = [(cust['id'], cust['begin_time']) for cust in route]
            routes_with_customers.append(route_with_customers)
        return routes_with_customers

    def plot_routes(self):
        no_of_vehicle = self.results["no_of_vehicle"]
        # Generate y coordinates for routes:
        y_list = [1/(no_of_vehicle + 1) * (i+1) for i in range(no_of_vehicle)]
        labels = ["vehicle {}".format(i+1) for i in range(no_of_vehicle)]
        # Plot intervals:
        count = 0
        for route in self.routes:
            y = y_list[count]
            count += 1
            for cust in route:
                self.__plot_line(y, cust['begin_time'], cust['begin_time'] + cust['duration'])
        plt.yticks(y_list, labels)
        plt.ylim(0,1)
        plt.show()

    def __plot_line(self, y, x_start, x_stop):
        plt.hlines(y, x_start, x_stop, lw = 10)
        plt.vlines(x_start, y-0.02, y+0.02, lw = 2)
        plt.vlines(x_stop, y-0.02, y+0.02, lw = 2)

    def __check_feasible(self,route):
        resource = sum([cust['resource'] for cust in route])
        if resource > self.fleet.capacity:
            print("Capacity not feasible")
