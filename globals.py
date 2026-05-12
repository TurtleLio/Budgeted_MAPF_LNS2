import re
import os
import math
import json
import time
import heapq
import random
import pstats
import cProfile
import itertools
from itertools import combinations, permutations, tee, pairwise
from datetime import datetime
from typing import *
from collections import deque, defaultdict

import numpy as np
import matplotlib.pyplot as plt


color_names = [
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'black',
    'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
    'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod',
    'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid',
    'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite',
    'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow',
    'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
    'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite',
    'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
    'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet',
    'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

markers = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "D",
    "d",
    "P",
    "X",
]
lines = [
    "-",
    "--",
    "-.",
    ":",
]

markers_iter = iter(markers)
markers_lines_dict = {}
colors_dict: DefaultDict[str, str | None] = defaultdict(lambda: None)



class Node:
    def __init__(self, x: int, y: int, neighbours: List[str] | None = None):
        self.x: int = x
        self.y: int = y
        self.neighbours: List[str] = [] if neighbours is None else neighbours
        self.neighbours_nodes: List[Node] = []
        self.xy_name: str = f'{self.x}_{self.y}'
        self._id: int | None = None
        self.heuristic_cache: Dict[str,int] = {}

    @property
    def xy(self):
        return self.x, self.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.xy_name < other.xy_name

    def __gt__(self, other):
        return self.xy_name > other.xy_name

    def __hash__(self):
        return hash(self.xy_name)

    def __str__(self):
        return self.xy_name

    def __repr__(self):
        return self.xy_name
    @property
    def id(self) -> int:
        """Fast integer ID - works for maps up to 1000x1000"""
        if self._id is None:
            self._id = self.x * 1000 + self.y
        return self._id

    def get_fast_heuristic(self, goal_xy_name: str, h_dict: Dict[str, np.ndarray]) -> int:
        """Cached heuristic lookup - much faster"""
        if goal_xy_name not in self._heuristic_cache:
            goal_h_dict = h_dict[goal_xy_name]
            self._heuristic_cache[goal_xy_name] = int(goal_h_dict[self.x, self.y])
        return self._heuristic_cache[goal_xy_name]

class AgentAlg:
    def __init__(self, num: int, start_node: Node, goal_node: Node):
        self.num = num
        self.name = f'agent_{num}'
        self.start_node: Node = start_node
        self.start_node_name: str = self.start_node.xy_name
        self.curr_node: Node = start_node
        self.goal_node: Node | None = goal_node
        self.goal_node_name: str = self.goal_node.xy_name
        self.alt_goal_node: Node | None = None
        self.message: str = ''
        self.path: List[Node] | None = [self.start_node]
        self.k_path: List[Node] | None = [self.start_node]
        self.temp_path: List[Node] | None = [self.start_node]
        self.init_priority: float = random.random()
        self.priority: float = self.init_priority
        self.best_path_without_constraints : List[Node] | None = [self.start_node]
        self.visited_count: int = 0
        self.starting_conflicts: int = 0
        self.current_path_length: int = 0
        self.k_path_delay: int = 0
        self.my_vertex_constraints: List[Tuple[int, int, int]] = []
        self.my_edge_constraints: List[Tuple[int, int, int, int, int]] = []
        self.my_permanent_constraint: Tuple[int, int] | None = None


    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    def update_curr_node(self, i_time):
        if i_time >= len(self.path):
            self.curr_node = self.path[-1]
            return
        self.curr_node = self.path[i_time]

    def get_goal_node(self) -> Node:
        if self.alt_goal_node is not None:
            return self.alt_goal_node
        if self.goal_node is not None:
            return self.goal_node
        return self.curr_node

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __lt__(self, other: Self):
        return self.priority < other.priority

    def __hash__(self):
        return hash(self.num)

    def __eq__(self, other):
        return self.num == other.num

def create_resource_class(base_distribution_class, neib_resources, neib_agents_distribution, class_name=None):
    if class_name is None:
        class_name = f"{base_distribution_class.distribution_name}-{neib_resources.neib_budget_name}-{neib_agents_distribution.neib_agents_name}"
    return type(class_name, (base_distribution_class,) + (neib_resources,) + (neib_agents_distribution,), {'class_name': class_name})

class resource_distribution_class:
    distribution_name: str
    def __init__(self, resources_per_agent=0, num_agents=0):
        self.max_nodes = 0
        self.resources_per_agent = resources_per_agent
        self.num_agents = num_agents
        self.distribution_name = None
        self.proportions = np.array([])
        self.agent_subset = []
        self.neib_budget = 0
        self.agent_distances = {}
        self.weights = {
            'Shared': {'value': 1, 'successes': 0, 'expansions': 0},
            'CPB': {'value': 1, 'successes': 0, 'expansions': 0},
            'RCPB': {'value': 1, 'successes': 0, 'expansions': 0},
            'PID': {'value': 1, 'successes': 0, 'expansions': 0}
        }
    def input_parameters(self,resources_per_agent: int, num_agents: int):
        self.resources_per_agent = resources_per_agent
        self.num_agents = num_agents
    def resource_distribuition(self, access_budget):
        pass
    def resource_subtraction(self,index):
        pass
    def resource_collection(self):
        self.proportions = np.array([])
        resource_sum = 0
        for i, nodes_left in enumerate(self.max_nodes):
            self.proportions= np.append(self.proportions, nodes_left)
            resource_sum += nodes_left
            self.max_nodes[i] = 0
        self.max_nodes = np.append(self.max_nodes,resource_sum)
        return self.proportions
    def resource_forward(self, agent : AgentAlg, mixed_agents_lis: List[AgentAlg]):
        return None

class fixed_distribution(resource_distribution_class):
    distribution_name = 'fixed'
    def __init__(self, resources_per_agent=0, num_agents=0):
        super().__init__(resources_per_agent, num_agents)
    def resource_distribuition(self, access_budget:int = 0):
        total_amount = np.array([self.resources_per_agent] * self.num_agents)
        j=0
        for i in range (access_budget):
            total_amount[j] = total_amount[j] + 1
            j += 1
            if j >= self.num_agents:
                j = 0
        self.max_nodes = total_amount
        return total_amount
    def resource_subtraction(self, index):
        if self.max_nodes[index] <= 0:
            return False
        self.max_nodes[index] -= 1
        return True
    def resource_forward(self, agent, mixed_agents_list):
        index = mixed_agents_list.index(agent)
        if index < (len(mixed_agents_list) - 1):
            next_agent = mixed_agents_list[index+1]
            self.max_nodes[next_agent.num] += self.max_nodes[agent.num]
            self.max_nodes[agent.num] = 0
        return None

class shared_distribution(resource_distribution_class):
    distribution_name = 'shared'
    def __init__(self, resources_per_agent=0, num_agents=0):
        super().__init__(resources_per_agent, num_agents)
    def resource_distribuition(self, access_budget:int = 0):
        total_amount = np.array([0] * self.num_agents)
        total_amount = np.append(total_amount,self.resources_per_agent* self.num_agents)
        j=0
        for i in range (access_budget):
            total_amount[j] = total_amount[j] + 1
            j += 1
            if j > self.num_agents:
                j = 0
        self.max_nodes = total_amount
        return total_amount
    def resource_subtraction(self, index):
        if self.max_nodes[-1] <= 0:
            return False
        self.max_nodes[-1] -= 1
        return True
    def resource_forward(self, agent, mixed_agents_list):
        return None


class PIB_distribution(resource_distribution_class):
    distribution_name = 'PIB'
    def __init__(self, resources_per_agent=0, num_agents=0):
        super().__init__(resources_per_agent, num_agents)
    def resource_distribuition(self, access_budget:int = 0):
        precentage_of_shared_pool = 50
        total_amount = self.resources_per_agent * self.num_agents
        shared_pool = total_amount * precentage_of_shared_pool / 100
        total_amount = np.array([self.resources_per_agent - int(shared_pool / self.num_agents)] * self.num_agents)
        total_amount = np.append(total_amount, shared_pool)
        j=0
        for i in range (access_budget):
            total_amount[j] = total_amount[j] + 1
            j += 1
            if j > self.num_agents:
                j = 0
        self.max_nodes = total_amount
        return total_amount
    def resource_subtraction(self, index):
        if self.max_nodes[-1] <= 0:
            return False
        if self.max_nodes[index] != 0:
            self.max_nodes[index] -= 1
        else:
            self.max_nodes[-1] -= 1
        return True


class neib_resources:
    neib_budget_name: str
    def neib_pool(self,neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5, pid: [float]=None):
        pass

class neib_sum(neib_resources):
    neib_budget_name = "Sum"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5, pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        total_amount = 0
        for agent in neib_agents:
            total_amount += self.proportions[agent.num]
        if total_amount > self.max_nodes[-1]:
            total_amount = self.max_nodes[-1]
        self.neib_budget = total_amount
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_proportions(neib_resources):
    neib_budget_name = "neighbourhood-proportions"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5, pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        self.proportions = [0]*self.num_agents
        sum_proportions = 0
        for agent, connections in cp_graph.items():
            agent_id = int(agent.split('_')[1])
            if len(connections) > 0 and len(connections) is not None:
                self.proportions[agent_id] = len(connections)
                sum_proportions += len(connections)
            else:
                self.proportions[agent_id] = 0.01
        self.proportions = tuple(self.proportions)
        for agent in neib_agents:
            self.neib_budget += int(self.max_nodes[-1]*self.proportions[agent.num]/sum(self.proportions))
        if self.neib_budget <= len(self.agent_subset):
            self.neib_budget = len(self.agent_subset)
        if self.neib_budget > self.max_nodes[-1]:
            self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_proportions_with_prefix(neib_resources):
    neib_budget_name = "Conflict-Proportional-Budget"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5, pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        prefix_sub_limit = prefix*10
        self.proportions = [0]*self.num_agents
        sum_proportions = 0
        for agent, connections in cp_graph.items():
            agent_id = int(agent.split('_')[1])
            if len(connections) > 0 and len(connections) is not None:
                self.proportions[agent_id] = len(connections)
                sum_proportions += len(connections)
            else:
                self.proportions[agent_id] = 0.01
        for agent in neib_agents:
            self.neib_budget += int(self.max_nodes[-1]*self.proportions[agent.num]/sum(self.proportions))
        if self.neib_budget <= len(self.agent_subset):
            self.neib_budget = len(self.agent_subset)
        if self.neib_budget < prefix_sub_limit:
            self.neib_budget = prefix_sub_limit
        if self.neib_budget > self.max_nodes[-1]:
            self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_proportions_with_prefix_reversed(neib_resources):
    neib_budget_name = "Reversed-Conflict-Proportional-Budget"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5, pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        prefix_sub_limit = prefix*10
        self.proportions = [0]*self.num_agents
        sum_proportions = 0
        for agent, connections in cp_graph.items():
            agent_id = int(agent.split('_')[1])
            if len(connections) > 0 and len(connections) is not None:
                self.proportions[agent_id] = len(connections)
                sum_proportions += len(connections)
            else:
                self.proportions[agent_id] = 0
        sum_reverse_proportions = sum(self.proportions)/2
        sum_conf_reverse = 0
        for agent in neib_agents:
            sum_conf_reverse += sum_reverse_proportions - self.proportions[agent.num]
        self.neib_budget =int(self.max_nodes[-1]*sum_conf_reverse / (sum_reverse_proportions*len(self.agent_subset)))
        if self.neib_budget <= len(self.agent_subset):
            self.neib_budget = len(self.agent_subset)
        if self.neib_budget < prefix_sub_limit:
            self.neib_budget = prefix_sub_limit
        if self.neib_budget > self.max_nodes[-1]:
            self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_distance_proportions(neib_resources):
    neib_budget_name = "Distance-Proportional-Budget"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5, pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        prefix_sub_limit = prefix*16
        sum_proportion = 0
        for agent in neib_agents:
            sum_proportion += self.agent_distances[agent.num]['normalized']
        self.neib_budget = self.max_nodes[-1] * sum_proportion/len(self.agent_subset)
        if self.neib_budget < prefix_sub_limit:
            self.neib_budget = prefix_sub_limit
        if self.neib_budget > self.max_nodes[-1]:
            self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_multi_arm_bandit(neib_resources):
    neib_budget_name = "Multi-Arm-Bandit"
    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 50, cp_graph: dict = [str, List[AgentAlg]],
                  prefix: int = 5, pid: List[int] = [0, 0, 0]):
        self.agent_subset = neib_agents
        return

class neib_random(neib_resources):
    neib_budget_name = "Random"

    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 50, cp_graph: dict = [str, List[AgentAlg]],
                  prefix: int = 5, pid: List[int] = [0, 0, 0]):
        self.agent_subset = neib_agents
        random_num = np.random.normal(0.5, 0.2)
        clamped_num = np.clip(random_num, 0, 1)
        self.neib_budget = int(self.max_nodes[-1]*clamped_num)
        if self.neib_budget < 20:
            self.neib_budget = 20
        if self.neib_budget > self.max_nodes[-1]:
            self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_random(neib_resources):
    neib_budget_name = "PIBT"

    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 50, cp_graph: dict = [str, List[AgentAlg]],
                  prefix: int = 5, pid: List[int] = [0, 0, 0]):
        self.agent_subset = neib_agents
        random_num = np.random.normal(0.5, 0.2)
        clamped_num = np.clip(random_num, 0, 1)
        self.neib_budget = int(self.max_nodes[-1]*clamped_num)
        if self.neib_budget < 20:
            self.neib_budget = 20
        if self.neib_budget > self.max_nodes[-1]:
            self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_fixed_50(neib_resources):
    neib_budget_name = "Fixed-50"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5,pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        if self.max_nodes[-1] < fixed_neib_nodes:
            fixed_neib_nodes = self.max_nodes[-1]
        self.neib_budget = fixed_neib_nodes
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_fixed_100(neib_resources):
    neib_budget_name = "Fixed-100"
    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 100,
                  cp_graph: dict = [str, List[AgentAlg]], prefix: int = 5, pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        if self.max_nodes[-1] < fixed_neib_nodes:
            fixed_neib_nodes = self.max_nodes[-1]
        self.neib_budget = fixed_neib_nodes
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_fixed_150(neib_resources):
    neib_budget_name = "neighbourhood-fixed-150"

    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 150,
                  cp_graph: dict = [str, List[AgentAlg]], prefix: int = 5, pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        if self.max_nodes[-1] < fixed_neib_nodes:
            fixed_neib_nodes = self.max_nodes[-1]
        self.neib_budget = fixed_neib_nodes
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_fixed_prefix(neib_resources):
    neib_budget_name = "neighbourhood-fixed-prefix"
    len_cp_grap = 0
    fixed_amount = 0
    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 150,
                  cp_graph: dict = [str, List[AgentAlg]], prefix: int =5, pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        try:
            self.fixed_amount = prefix*16
            self.fixed_amount = tuple(self.fixed_amount)
        except:
            pass
        if self.max_nodes[-1] < self.fixed_amount:
            self.neib_budget = self.max_nodes[-1]
        else:
            self.neib_budget = self.fixed_amount
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_shared(neib_resources):
    neib_budget_name = "Shared"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]],prefix: int = 5, pid: List[int] = [0,0,0]):
        self.agent_subset = neib_agents
        self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

class neib_agents_distribution:
    neib_agents_name: str
    def agent_distribution(self):
        pass
    def neib_resource_subtraction(self,neib_agents:[AgentAlg],index):
        pass
    def return_resources(self):
        pass

class neib_agents_shared(neib_agents_distribution):
    neib_agents_name = "agents-shared"
    def agent_distribution(self):
        total_sum = 0
        for agent in self.agent_subset:
            self.max_nodes[agent.num] = int(self.neib_budget/len(self.agent_subset))
            total_sum += self.max_nodes[agent.num]
        self.neib_budget -= total_sum
        if total_sum <= 0:
            self.neib_budget = 0
        return np.array([self.max_nodes[agent.num] for agent in self.agent_subset])
    def neib_resource_subtraction(self, agent_num):
        index_list = []
        for agent in self.agent_subset:
            index_list.append(agent.num)
        for index in index_list:
            if self.max_nodes[index] > 0:
                self.max_nodes[index] -= 1
                return True
        return False
    def return_resources(self):
        returning_resources = sum(np.array([self.max_nodes[agent.num] for agent in self.agent_subset])) + self.neib_budget
        for agent in self.agent_subset:
            self.neib_budget += self.max_nodes[agent.num]
        for agent in self.agent_subset:
            self.max_nodes[agent.num] = 0
        self.max_nodes[-1] += self.neib_budget
        self.neib_budget = 0
        return returning_resources

class neib_agents_evenly_split(neib_agents_distribution):
    neib_agents_name = "agents-evenly-split"
    def agent_distribution(self):
        total_sum = 0
        for agent in self.agent_subset:
            self.max_nodes[agent.num] = int(self.neib_budget/len(self.agent_subset))
            total_sum += self.max_nodes[agent.num]
        self.neib_budget -= total_sum
        if total_sum <= 0:
            self.neib_budget = 0
        return np.array([self.max_nodes[agent.num] for agent in self.agent_subset])
    def neib_resource_subtraction(self, agent_num):
        index_list = []
        if self.max_nodes[agent_num] > 0:
            self.max_nodes[agent_num] -= 1
            return True
        return False
    def return_resources(self):
        total_amount = 0
        for agent in self.agent_subset:
            self.neib_budget += self.max_nodes[agent.num]
        for agent in self.agent_subset:
            try:
                self.proportions[agent.num] = int(self.neib_budget/len(self.agent_subset))
            except TypeError:
                pass
            self.max_nodes[agent.num] = 0
        self.max_nodes[-1] += self.neib_budget
        returning_resources = self.neib_budget
        self.neib_budget =0
        return returning_resources

class neib_agents_portion(neib_agents_distribution):
    neib_agents_name = "agent-propotion"
    def agent_distribution(self):
        total_sum = 0
        sum_proportions = 0
        for agent in self.agent_subset:
            sum_proportions += self.proportions[agent.num]
        for agent in self.agent_subset:
            self.max_nodes[agent.num] = int(self.neib_budget*self.proportions[agent.num]/sum_proportions)
            total_sum += self.max_nodes[agent.num]
        self.neib_budget -= total_sum
        if total_sum <= 0:
            self.neib_budget = 0
        return np.array([self.max_nodes[agent.num] for agent in self.agents_subset])
    def neib_resource_subtraction(self, agent_num):
        index_list = []
        if self.max_nodes[agent_num] > 0:
            self.max_nodes[agent_num] -= 1
            return True
        return False
    def return_resources(self):
        total_amount = 0
        for agent in self.agent_subset:
            total_amount += self.max_nodes[agent.num]
            if self.neib_budget_name != "neighbourhood-proportions":
                try:
                    self.proportions[agent.num] = self.max_nodes[agent.num]
                except TypeError:
                    pass
        for agent in self.agent_subset:
            self.max_nodes[agent.num] = 0
        self.max_nodes[-1] += total_amount
        return self.max_nodes[-1]

class neib_pid(neib_resources):
    neib_budget_name = "Pid"

    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 150,
                  cp_graph: dict = [str, List[AgentAlg]], prefix: int = 5, pid: List[float] = None):
        kp, kd, ki = pid[0], pid[1], pid[2]
        budget_sum = 0
        self.agent_subset = neib_agents
        for agent in self.agent_subset:
            agent.visited_count += 1
            agent_budget = self.calculate_budget_pid(agent, kp, kd, ki, prefix, cp_graph)
            budget_sum += agent_budget
        if self.max_nodes[-1] < budget_sum:
            budget_sum = self.max_nodes[-1]
        self.neib_budget = int(budget_sum)
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

    def calculate_budget_pid(self, agent, kp, kd, ki, prefix, cp_graph: Dict[str, List[AgentAlg]]):
        conflict_count = len(cp_graph.get(agent.name, []))
        proportional = conflict_count
        differential = agent.starting_conflicts - conflict_count
        integral = agent.visited_count
        if differential < 0:
            differential = 0

        agent_budget = kp * prefix / (proportional+1) + kd * prefix * differential + ki * prefix * integral
        return agent_budget

class neib_pid_distance(neib_resources):
    neib_budget_name = "Pid-Distance"

    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 150,
                  cp_graph: dict = [str, List[AgentAlg]], prefix: int = 5, pid: List[float] = [1.5,1.75,0.5]):
        kp, kd, ki = pid[0], pid[1], pid[2]
        budget_sum = 0
        self.agent_subset = neib_agents
        for agent in self.agent_subset:
            agent.visited_count += 1
            agent_budget = self.calculate_budget_pid_delays(agent, kp, kd, ki, prefix)
            budget_sum += agent_budget
        if self.max_nodes[-1] < budget_sum:
            budget_sum = self.max_nodes[-1]
        if budget_sum < 0:
            budget_sum = 0
        self.neib_budget = int(budget_sum)
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

    def calculate_budget_pid_delays(self, agent, kp, kd, ki, prefix):
        proportional = self.agent_distances[agent.num]['normalized']
        differential = abs(agent.current_path_length - self.agent_distances[agent.num]['raw'])
        integral = agent.visited_count
        agent_budget = kp * prefix * proportional + kd * prefix * differential + ki * prefix * integral
        return agent_budget

class neib_pid_pie(neib_resources):
    neib_budget_name = "Pid-Pie"

    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 150,
                  cp_graph: dict = [str, List[AgentAlg]], prefix: int = 5, pid: List[float] = None):
        kp, kd, ki = pid[0], pid[1], pid[2]
        budget_sum = 0
        self.agent_subset = neib_agents
        for agent in self.agent_subset:
            agent.visited_count += 1
            agent_budget = self.calculate_budget_pid_pie(agent, kp, kd, ki, prefix)
            budget_sum += agent_budget
        if self.max_nodes[-1] < budget_sum:
            budget_sum = self.max_nodes[-1]
        if budget_sum < 0:
            budget_sum = 0
        self.neib_budget = int(budget_sum)
        self.max_nodes[-1] -= self.neib_budget
        return self.neib_budget

    def calculate_budget_pid_pie(self, agent, kp, kd, ki, prefix):
        manhattan_dist = abs(agent.goal_node.x - agent.curr_node.x) + abs(agent.goal_node.y - agent.curr_node.y)
        proportional = agent.current_path_length - manhattan_dist
        differential = agent.starting_conflicts - agent.current_path_length
        integral = agent.visited_count
        if differential < 0:
            differential = 0
        agent_budget = kp * prefix * proportional + kd * prefix * differential + ki * prefix * integral
        agent.starting_conflicts = agent.current_path_length
        return agent_budget


class resource_and_neib_distributions:
    @staticmethod
    def create(distribution_class, neighbor_class=None, neighbor_agents_class=None, resources_per_agent=0, num_agents=0):
        DynamicClass = create_resource_class(distribution_class, neighbor_class, neighbor_agents_class)
        return DynamicClass(resources_per_agent=resources_per_agent, num_agents=num_agents)
