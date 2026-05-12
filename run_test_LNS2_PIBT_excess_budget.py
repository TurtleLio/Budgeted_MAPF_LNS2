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
    # Iterate over direct subclasses of resource_distribution_class
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
    alg_name = sys.argv[3]  # oneshot_LNS2 / lifelong_LNS2
    resource_distribution_name = sys.argv[4]
    neighbourhood_distribution_name = sys.argv[5]
    agents_distribution_name = sys.argv[6]
    resources_per_agent = int(sys.argv[7])
    prefix = int(sys.argv[8])
    folder_name = sys.argv[9]
    #scene_index = int(sys.argv[10])
    pid = [1,1.5,1]
    if 'PIBT' not in neighbourhood_distribution_name:
        resource_distribution = get_distribution_class(resource_distribution_name)
        neighbourhood_distribution = get_distribution_class(neighbourhood_distribution_name)
        agents_distribution = get_distribution_class(agents_distribution_name)
    else:
        resource_distribution = get_distribution_class(resource_distribution_name)
        neighbourhood_distribution = get_distribution_class('PIBT')
        agents_distribution = get_distribution_class(agents_distribution_name)

    # results_dir = "../results/PIBT+/number_of_agents"
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
    # results_dir = "../results/PIBT+/resources_per_agent"
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
    # results_dir = "../results/PIBT+/prefix"
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
    #
    # for scene_index in range(1,26):
    #         main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid,folder_name, scene_index)
    #         print(f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {resources_per_agent} | {prefix} | {scene_index}")
    # main(map_name, number_of_agents, alg_name, resource_distribution_name, neighbourhood_distribution, agents_distribution, resources_per_agent, scene_index)

    # specific_distributions = [
    #     {"resource_distribution": "fixed",
    #      "neighbourhood_distribution": "Shared",
    #      "agents_distribution": "agents-shared"},
    #     {"resource_distribution": "fixed",
    #      "neighbourhood_distribution": "Conflict-Proportional-Budget",
    #      "agents_distribution": "agents-shared"},
    #     {"resource_distribution": "fixed",
    #      "neighbourhood_distribution": "Reversed-Conflict-Proportional-Budget",
    #      "agents_distribution": "agents-shared"},
    #     {"resource_distribution": "fixed",
    #      "neighbourhood_distribution": "Pid",
    #      "agents_distribution": "agents-shared"},
    #     {"resource_distribution": "fixed",
    #      "neighbourhood_distribution": "Multi-Arm-Bandit",
    #      "agents_distribution": "agents-shared"},
    # ]
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
    # maps = ['random-32-32-10.map','room-32-32-4.map','maze-32-32-4.map']
    # resources_array = [5,10,15,8, 14, 20, 50, 100, 200, 11, 25, 80, 150, 250, 300, 350, 400]
    resources_array = [5,10,15,20, 50, 100, 200, 25, 80, 150, 250, 300, 350, 400]
    prefix_array = [5,10,15,20,25,30,35]
    # resources_array = [50, 200]
    # for specific_distribution in specific_distributions:
    #     resource_distribution = get_distribution_class(specific_distribution['resource_distribution'])
    #     neighbourhood_distribution = get_distribution_class(specific_distribution['neighbourhood_distribution'])
    #     agents_distribution = get_distribution_class(specific_distribution['agents_distribution'])
    #     # for scene_index in range(1, 26):
    #     #     main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution, agents_distribution,
    #     #          resources_per_agent, prefix, pid, folder_name, scene_index)
    #     #     print(
    #     #         f"done {map_name} | {number_of_agents} | {alg_name} | {specific_distribution['resource_distribution']} | {specific_distribution['neighbourhood_distribution']} | {specific_distribution['agents_distribution']} | {resources_per_agent} | {prefix} | {scene_index}")
    #
    # # for resources_per_agent in resources_array:
    # #     for scene_index in range(1, 26):
    # #         main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid,folder_name, scene_index)
    # #         print(f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {scene_index}")
    #     for resources_per_agent in resources_array:
    #         main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution,
    #              agents_distribution, resources_per_agent, prefix, pid, folder_name, scene_index)
    #         print(
    #             f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {scene_index}")
    # # for map in maps:
    #     number_of_agents = fixed_agents_by_map[map]
    #     for resources_per_agent in resources_array:
    #         main(map, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution,
    #              agents_distribution, resources_per_agent, prefix, pid, folder_name, scene_index)
    #         print(
    #             f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {scene_index}")

    # for specific_distribution in specific_distributions:
    #     resource_distribution = get_distribution_class(specific_distribution['resource_distribution'])
    #     neighbourhood_distribution = get_distribution_class(specific_distribution['neighbourhood_distribution'])
    #     agents_distribution = get_distribution_class(specific_distribution['agents_distribution'])
    #     for resources_per_agent in resources_array:
    #         main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid,folder_name, scene_index)
    #         print(f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {scene_index}")
    # #     # for resources_per_agent in resources_array:
    # #     #     main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution, agents_distribution, resources_per_agent,prefix,pid,folder_name, scene_index)
    # #     #     print(f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {scene_index}")
    #     main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution, agents_distribution,
    #          resources_per_agent, prefix, pid, folder_name, scene_index)
    #     print(
    #     f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {scene_index}")
    # main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution, agents_distribution,
    #      resources_per_agent, prefix, pid, folder_name, scene_index)
    # print(
    # f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {scene_index}")
    # for resources_per_agent in resources_array:
    #     main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution,
    #          agents_distribution, resources_per_agent, prefix, pid, folder_name, scene_index)
    #     print(
    #         f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {scene_index}")
    for i in range (1,26):
        main(map_name, number_of_agents, alg_name, resource_distribution, neighbourhood_distribution,
             agents_distribution, resources_per_agent, prefix, pid, folder_name, i)
        print(
            f"done {map_name} | {number_of_agents} | {alg_name} | {resource_distribution_name} | {neighbourhood_distribution_name} | {agents_distribution_name} | {resources_per_agent} | {prefix} | {i}")

