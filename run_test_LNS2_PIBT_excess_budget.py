import sys
from algs.alg_choise_LNS2_PIBT_excess_budget import run_best_of_excess_budget
from globals import *
import matplotlib
import numpy as np

def main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid,folder_name,  scene_index):
    if alg_name == 'oneshot_LNS2_PIBT_LNS1':
        run_best_of_excess_budget(map_name, number_of_agents, 'k-LNS2-PIBT-LNS1', resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid, folder_name, scene_index)
    if alg_name == 'lifelong_LNS2_PIBT_LNS1':
        run_best_of_excess_budget(map_name, number_of_agents, 'Lifelong-LNS2-PIBT-LNS1', resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid, folder_name, scene_index)
    if alg_name == 'oneshot_LNS2_PIBT_spillover':
        run_best_of_excess_budget(map_name, number_of_agents, 'k-LNS2-PIBT-spillover', resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid, folder_name, scene_index)
    if alg_name == 'lifelong_LNS2_PIBT_spillover':
        run_best_of_excess_budget(map_name, number_of_agents, 'Lifelong-LNS2-PIBT-spillover', resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid, folder_name, scene_index)
    if alg_name == 'oneshot_LNS2_PIBT_LNS1-pid':
        run_best_of_excess_budget(map_name, number_of_agents, 'k-LNS2-PIBT-LNS1-pid', resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid, folder_name, scene_index)
    if alg_name == 'lifelong_LNS2_PIBT_LNS1-pid':
        run_best_of_excess_budget(map_name, number_of_agents, 'Lifelong-LNS2-PIBT-LNS1-pid', resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid, folder_name, scene_index)
    if alg_name == 'oneshot_LNS2_PIBT_LNS1-DPB':
        run_best_of_excess_budget(map_name, number_of_agents, 'k-LNS2-PIBT-LNS1-DPB', resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid, folder_name, scene_index)
    if alg_name == 'lifelong_LNS2_PIBT_LNS1-DPB':
        run_best_of_excess_budget(map_name, number_of_agents, 'Lifelong-LNS2-PIBT-LNS1-DPB', resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid, folder_name, scene_index)
    if alg_name == 'oneshot_LNS2_PIBT':
        run_best_of_excess_budget(map_name, number_of_agents, 'k-LNS2-PIBT', resource_distribution,
                                  neighbourhood_distribution, agents_distribution, resources_per_agent, prefix, pid,
                                  folder_name, scene_index)
    if alg_name == 'lifelong_LNS2_PIBT':
        run_best_of_excess_budget(map_name, number_of_agents, 'Lifelong-LNS2-PIBT', resource_distribution,
                                  neighbourhood_distribution, agents_distribution, resources_per_agent, prefix, pid,
                                  folder_name, scene_index)


def get_distribution_class(name):
    for subclass in resource_distribution_class.__subclasses__():
        if getattr(subclass, 'distribution_name', None) == name:
            return subclass
    for subclass in neib_resources.__subclasses__():
        if getattr(subclass, 'neib_budget_name', None) == name:
            return subclass
    for subclass in neib_agents_distribution.__subclasses__():
        if getattr(subclass, 'neib_agents_name', None) == name:
            return subclass
    raise ValueError(f"No distribution class found with distribution_name: {name}")

if __name__ == '__main__':
    map_name = sys.argv[1]
    number_of_agents = int(sys.argv[2])
    alg_name = sys.argv[3]
    resource_distribution_name = sys.argv[4]
    neighbourhood_distribution_name = sys.argv[5]
    agents_distribution_name = sys.argv[6]
    resources_per_agent = int(sys.argv[7])
    prefix = int(sys.argv[8])
    folder_name = sys.argv[9]
    pid = [1,1.5,1]
    if 'PIBT' not in neighbourhood_distribution_name:
        resource_distribution = get_distribution_class(resource_distribution_name)
        neighbourhood_distribution = get_distribution_class(neighbourhood_distribution_name)
        agents_distribution = get_distribution_class(agents_distribution_name)
    else:
        resource_distribution = get_distribution_class(resource_distribution_name)
        neighbourhood_distribution = get_distribution_class('PIBT')
        agents_distribution = get_distribution_class(agents_distribution_name)


    specific_distributions = [
            {"resource_distribution": "fixed",
             "neighbourhood_distribution": "Pid",
             "agents_distribution": "agents-shared"}
    ]
    fixed_agents_by_map = {
        "random-32-32-10.map": 300,
        "empty-32-32.map": 340,
        "random-32-32-20.map": 110,
        "room-32-32-4.map": 120,
        "maze-32-32-2.map": 40,
        "maze-32-32-4.map": 30
    }
    resources_array = [5,10,15,20, 50, 100, 200, 25, 80, 150, 250, 300, 350, 400]
    prefix_array = [5,10,15,20,25,30,35]

    for i in range (1,26):
        main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution,
             agents_distribution, resources_per_agent, prefix, pid, folder_name, i)
        print(
            f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {i}")

