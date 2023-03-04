from data_utils import load_config_data, load_operation, load_flights, Map, excel_reader, excel_writer, get_tuple
from tws import tw_schedule
from vrptw import VRPTW
import copy
import datetime
from collections import namedtuple
import pickle

class Instance:
    def __init__(self, operation, map, flights):
        # For counting time
        self.start_time = datetime.datetime.now()

        self.operation = operation
        self.services = operation["services"]
        self.tasks = operation["tasks"]
        self.fleets = operation["fleets"]
        self.flights = flights
        self.distance = self.load_distance_matrix()

        self.solutions = []
        self.init_time_windows = {}

        self.excel_output = None

    def load_distance_matrix(self):
        # Get the list of locations, "depot" being the location of depot
        loc_list = [self.flights[id].location for id in range(1, len(self.flights)+1)]
        loc_list.insert(0, 'depot')
        loc_list.append('depot')
        distance = map.load_distance(loc_list)
        return [[distance[(i,j)] for j in loc_list] for i in loc_list]

    def solve(self):

        # Initial time window scheduling
        self.cal_init_tw()
        sol_data = (self.operation, self.distance, self.flights, self.init_time_windows)
        # Different solutions catagorized by diff order of sub-problems
        order = self.tasks ###########
        sol = Solution(sol_data, order)
        sol.solve()
        # finished = False
        # while not finished:
        #     # Generate order of solving sub-problems(VRPTW)
        #     order = [1] # test
        #     sol = Solution(sol_data, order)
        #     sol.solve()
        #     finished = True # test


        # result calculation ###########
        time_elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        total_distance = sum([sol.fleet_schedules[task].results["total_distance"] for task in self.tasks])
        vehicle_of_fleets = [sol.fleet_schedules[task].results["no_of_vehicle"] for task in self.tasks]
        total_vehicle = sum(vehicle_of_fleets)
        print("Final total:", total_distance)

        self.excel_output = [time_elapsed, total_distance, total_vehicle]
        self.excel_output.extend(vehicle_of_fleets)

    def cal_init_tw(self):
        # Calculate initial time windows of each flight without any planned tasks on any
        Task = namedtuple("Task", "duration precedence_before precedence_after exact_time")
        for id in self.flights:
            flight = self.flights[id]
            time_slot = (flight.arrival, flight.departure)
            tasks = {task: Task(
                    self.services[flight.type][task].duration,
                    self.services[flight.type][task].precedence_before,
                    self.services[flight.type][task].precedence_after,
                    None) for task in self.tasks}

            self.init_time_windows[id] = tw_schedule(time_slot, tasks)




class Solution():
    def __init__(self, sol_data, order):
        self.fleets = sol_data[0]["fleets"]
        self.services = sol_data[0]["services"]
        self.tasks = sol_data[0]["tasks"]
        self.types = sol_data[0]['types']
        self.distance = sol_data[1]
        self.flights = sol_data[2]
        self.init_time_windows = sol_data[3]
        self.order = order

        # initialize VRPTWs:
        self.fleet_schedules = self.__init_schedules()

        self.__time_windows = copy.deepcopy(self.init_time_windows)
    def __init_schedules(self):
        return {task: VRPTW(fleet = self.fleets[task], distance = self.distance) for task in self.tasks}

    def solve(self):
        # Solve sub-problems in order
        for task in self.order:

            # Solve the sub-problem

            # Generate customer data for VRPTW
            customers = {}
            for id in self.flights:
                flight = self.flights[id]

                Flight = namedtuple("Flight", "time_window duration resource")
                time_window = self.__time_windows[id][task]
                service_info = self.services[flight.type][task]
                customers[id] = Flight(time_window, service_info.duration, service_info.resource)
            # Solve VRPTW
            self.fleet_schedules[task].load_customers(customers)
            self.fleet_schedules[task].routing()
            # Print to screen
            print("======Vehicle {}======".format(task))
            print("Total distance: {}".format(self.fleet_schedules[task].results["total_distance"]))
            print("No of vehicle: {}".format(self.fleet_schedules[task].results["no_of_vehicle"]))

            # Update time windows
            # print(self.fleet_schedules[task].cust_begin_time)
            self.__update_tw(task, self.fleet_schedules[task].cust_begin_time)
            # print(self.__time_windows)

        # End


    def __update_tw(self, task_update, exact_time_update):
        # Calculate initial time windows of each flight without any planned tasks on any
        Task = namedtuple("Task", "duration precedence_before precedence_after exact_time")
        for id in self.flights:
            # print("id of flight: {}".format(id))
            flight = self.flights[id]
            time_slot = (flight.arrival, flight.departure)
            tasks = {task: Task(
                    self.services[flight.type][task].duration,
                    self.services[flight.type][task].precedence_before,
                    self.services[flight.type][task].precedence_after,
                    None) for task in self.tasks}
            tasks[task_update] = Task(
                    self.services[flight.type][task_update].duration,
                    self.services[flight.type][task_update].precedence_before,
                    self.services[flight.type][task_update].precedence_after,
                    exact_time_update[id])

            self.__time_windows[id] = tw_schedule(time_slot, tasks)
            # print("old:", self.init_time_windows[id])
            # print("new:", self.__time_windows[id])


if __name__ == '__main__':
    # Load operation and map data:
    dir = 'data/'
    operation_dir = dir + 'config/operation_W.xlsx'
    net_dir = dir + 'config/airport.net.xml'
    cfg_dir = dir + 'config/testPrj.cfg'

    operation, map = load_config_data(operation_dir, net_dir, cfg_dir)

    # data to be written to excel:
    
    fleet_names = [operation["fleets"][task].name for task in operation["tasks"]]
    data = {}
    column_names = ["Instance", "Time(s)","Distance (m)", "Total number of vehicle"]
    column_names.extend(fleet_names)

    # Experiment configuration: (no of flights in an instance, no of instance)
    # possible number of flights: 20, 50, 100, 200, 300
    experiments = [ (100, 10), (200, 10)]
    # experiments = [(20, 10)]

    for setting in experiments:
        no_of_flights = setting[0]
        no_of_instance = setting[1]
        # Add a new sheet with sheet name = column_names to excel
        sheet_name = "Flight " + str(no_of_flights)
        data[sheet_name] = [column_names]
        print("=====Flight {}=====".format(no_of_flights))

        for i in range(no_of_instance):
            print("Solving instance [{}/{}] ...".format(i + 1, no_of_instance))
            instance_path = dir + 'instance/'+ str(no_of_flights) + '/schedule_{}.xlsx'.format(i+1)
            data[sheet_name].append(["Instance {}".format(i + 1)])

            # Load flight info
            flights = load_flights(instance_path)
            # Create problem instance
            instance = Instance(operation, map, flights)
            instance.solve()

            data[sheet_name][-1].extend(instance.excel_output)###########
            print("Solved.")

            # save data every 10 instance

            if i != 0 and i % 10 == 0:
                data_setting = copy.deepcopy(data)
                get_tuple(data_setting)
                out_dir = 'log/' + str(datetime.datetime.now()) + "_" + str(no_of_flights) + "_" + str(i//10) + '.xlsx'
                excel_writer(data_setting, out_dir)


        data_setting = copy.deepcopy(data)
        get_tuple(data_setting)
        out_dir = 'log/' + str(datetime.datetime.now()) + "_" + str(no_of_flights) + '.xlsx'
        excel_writer(data_setting, out_dir)



    get_tuple(data)
    out_dir = 'log/' + str(datetime.datetime.now()) + '.xlsx'
    excel_writer(data, out_dir)
