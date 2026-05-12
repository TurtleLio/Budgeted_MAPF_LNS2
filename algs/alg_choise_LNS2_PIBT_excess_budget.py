import math
import matplotlib
import numpy as np
from globals import *
from algs.alg_functions_LNS2 import *
from algs.alg_sipps import run_sipps
from run_single_MAPF_func import run_mapf_alg
from algs.real_alg_functions_pibt import *
from algs.alg_prefix_PIBT import *
from algs.alg_functions_PIE import *
import copy


def compute_distance(current_node, goal_node, h_dict):

    goal_heuristic_map = h_dict[goal_node.xy_name]

    dist = int(goal_heuristic_map[current_node.x, current_node.y])

    return dist



def calculate_state_score(LNS_agents, pibt_agents, choise_counter, h_dict):
    def sorted_distance_array(agents,h_dict):
        distances = []
        for agent in agents:
            if agent.goal_node is not None and agent.path:
                d = compute_distance(agent.k_path[-1], agent.goal_node,h_dict)
                distances.append(d)
        distances.sort(reverse=True)
        return distances

    def sorted_arival_time_array(agents):
        distances =[]
        for agent in agents:
            d = compute_arival_time(agent.k_path, agent.goal_node)
            distances.append(d)
        distances.sort(reverse=True)
        return distances

    distances_LNS = sorted_distance_array(LNS_agents,h_dict)
    distances_pibt = sorted_distance_array(pibt_agents,h_dict)
    if np.all(np.array(distances_pibt) == 0) and np.all(np.array(distances_LNS) == 0):
        distances_LNS = sorted_arival_time_array(LNS_agents)
        distances_pibt = sorted_arival_time_array(pibt_agents)

    for LNS_dist, PIBT_dist in zip(distances_LNS, distances_pibt):
        if LNS_dist != PIBT_dist:
            if LNS_dist < PIBT_dist:
                choise_counter[0] += 1
                return LNS_agents
            else:
                choise_counter[1] += 1
                return pibt_agents
    choise_counter[0] += 1
    return LNS_agents


def turn_pibt_path_to_k_path(agents: List[AgentAlg], step_iter:int =0,k_limit:int =0)->None:
    for agent in agents:
        agent.k_path = agent.path[:k_limit+1]
        agent.path = [agent.path[0]]

def compute_arival_time(path:List[Node], goal:Node)->int:
    counter = 0
    for node in path:
        if node != goal:
            counter += 1
        else:
            break
    return counter

def evaluate_output_state(algorithm_output):
    """
    Given the output state of your MAPF algorithm (usually a tuple with a dictionary
    of agent paths and an info dictionary containing the agents list),
    this function returns the overall state score.

    For example, if your algorithm returns (paths, info) and info contains:
        {'agents': [agent1, agent2, ...], ...}
    we pass the list of agents to the scoring function.
    """
    _, info = algorithm_output
    agents = info.get('agents', [])
    return calculate_state_score(agents)

from algs.alg_functions_LNS2 import *
from algs.alg_sipps import run_sipps
from algs.alg_limited_temporal_a_star_neighb import *
from run_single_MAPF_func import run_mapf_alg
import matplotlib

def solve_k_LNS2(
        agents: List[AgentAlg],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        vc_empty_np, ec_empty_np, pc_empty_np,
        params: dict,
        step_iter: int,
        excess_budget,
        heuristic_counter: Dict[str, int],
) -> int:
    alg_name: bool = params['alg_name']
    k_limit: int = params['k_limit']
    n_neighbourhood: bool = params['n_neighbourhood']
    max_iter_time: int = params['max_iter_time']
    pf_alg = params['pf_alg']
    pf_alg_name: str = params['pf_alg_name']
    resources = params['distribution_function']
    resources_per_agent = params['resources_per_agent']
    max_depth_search = params['max_depth_search']
    pid = params['pid']
    number_of_agents = params['number_of_agents']
    iter_start_time = time.time()
    tabu_list = []
    heuristic_weights = {'agent_based':1,'map_based':1,'random':1}
    agents = get_shuffled_agents(agents)

    resources.resource_distribuition(excess_budget)
    create_k_limit_init_solution(
        agents, nodes, nodes_dict, h_dict, map_dim, pf_alg_name, pf_alg, k_limit, iter_start_time,
        vc_empty_np, ec_empty_np, pc_empty_np, resources, max_depth_search, params
    )

    cp_graph, cp_graph_names = get_k_limit_cp_graph(agents, k_limit=k_limit)
    cp_len = len(cp_graph)
    occupied_from: Dict[str, AgentAlg] = {a.curr_node.xy_name: a for a in agents}
    resources.resource_collection()
    agent_distances = {
        agent_id: {'raw': 0.0, 'normalized': 0.0}
        for agent_id in range(len(agents))
    }
    for i in range(len(agents)):
        agent_id = agents[i].num
        new_dist = calculate_delay(agents[i].k_path[-1], agents[i].goal_node, h_dict)
        agent_distances[agent_id]['raw'] = new_dist
    lns_iter = 0
    subset_index = 0
    for agent_name in cp_graph.keys():
        idx = int(agent_name.split('_')[1])
        agents[idx].starting_conflicts = len(cp_graph[agent_name])
    while cp_len > 0 and has_resources_in_cp_graph(cp_graph, resources, agents):
        normalize_agent_distances(agent_distances)
        resources.agent_distances = agent_distances

        lns_iter += 1
        runtime = time.time() - iter_start_time
        agents_subset: List[AgentAlg] = get_k_limit_agents_subset(
            cp_graph, cp_graph_names, n_neighbourhood, agents, occupied_from, h_dict, resources
        )
        old_paths: Dict[str, List[Node]] = {a.name: a.k_path[:] for a in agents_subset}
        agents_outer: List[AgentAlg] = [a for a in agents if a not in agents_subset]

        solve_k_limit_subset_with_prp(
            agents_subset, agents_outer, nodes, nodes_dict, h_dict, map_dim, iter_start_time,
            pf_alg_name, pf_alg, vc_empty_np, ec_empty_np, pc_empty_np, resources, max_depth_search, k_limit, agents, cp_graph, pid,params['distribution_name'],heuristic_counter, params['rng']
        )
        old_cp_graph, old_cp_graph_names = cp_graph, cp_graph_names
        cp_graph, cp_graph_names = get_k_limit_cp_graph(agents_subset, agents_outer, cp_graph, k_limit=k_limit)
        cp_len = len(cp_graph)
        for agent in agents_subset:
            new_dist = calculate_delay(agent.k_path[k_limit],agent.goal_node, h_dict)
            agent_distances[agent.num]['raw'] = new_dist
    repair_agents_k_paths(agents, k_limit)
    excess_budget = resources.max_nodes[-1]
    return excess_budget

def improve_LNS1_shared(        agents: List[AgentAlg],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: dict,
        step_iter: int,
        excess_budget: int = 0,) -> None:
    alg_name: bool = params['alg_name']
    k_limit: int = params['k_limit']
    n_neighbourhood: bool = params['n_neighbourhood']
    max_iter_time: int = params['max_iter_time']
    pf_alg = params['pf_alg']
    pf_alg_name: str = params['pf_alg_name']
    resources = params['distribution_function']
    resources_per_agent = params['resources_per_agent']
    max_depth_search = params['max_depth_search']
    iter_start_time = time.time()
    number_of_agents = params['number_of_agents']
    tabu_list = []
    pid = params['pid']
    heuristic_weights = {'agent_based': 1, 'map_based': 1, 'random': 1}
    all_goals_reached = False
    agent_distances = {
        agent_id: {'raw': 0.0, 'normalized': 0.0}
        for agent_id in range(len(agents))
    }
    for i in range(len(agents)):
        agent_id = agents[i].num
        new_dist = calculate_delay(agents[i].k_path[-1], agents[i].goal_node, h_dict)
        agent_distances[agent_id]['raw'] = new_dist
        agents[i].current_path_length = new_dist
    if all(d["raw"] == 0 for d in agent_distances.values()):
        all_goals_reached = True
        for i in range(len(agents)):
            agent_id = agents[i].num
            new_dist = calculate_arrival_time_single_agent(agents[i])
            agent_distances[agent_id]['raw'] = new_dist
            agents[i].current_path_length = new_dist
    vc_hard_np, ec_hard_np, pc_hard_np = init_constraints_pie(map_dim, params['n_steps'] + 1)
    for agent in agents:
        if len(agent.k_path) != 1:
            update_constraints_tracked(agent.path+agent.k_path[1:k_limit+1], vc_hard_np, ec_hard_np, pc_hard_np, agent, 0)
        else:
            update_constraints_tracked(agent.path, vc_hard_np, ec_hard_np, pc_hard_np,
                                       agent, 0)
    if 'pid' in params['alg_name']:
        resources_LNS1 = resource_and_neib_distributions.create(shared_distribution, neib_pid_distance,
                                                           neib_agents_shared, 0, number_of_agents)
    elif 'DPB' in params['alg_name']:
        resources_LNS1 = resource_and_neib_distributions.create(shared_distribution, neib_distance_proportions,
                                                           neib_agents_shared, 0, number_of_agents)
    else:
        resources_LNS1 = resource_and_neib_distributions.create(shared_distribution, neib_shared,
                                                           neib_agents_shared, 0, number_of_agents)
    resources_LNS1.resource_distribuition(excess_budget)
    resources_LNS1.resource_collection()
    lns1_iter = 0
    old_resources = 0
    improved = False
    while resources_LNS1.max_nodes[-1] > 0:
        if len(tabu_list) == len(agents):
            tabu_list = []
        max_path_len = max([len(agent.path) for agent in agents])
        normalize_agent_distances(agent_distances)
        resources_LNS1.agent_distances = agent_distances
        lns1_iter += 1
        runtime = time.time() - iter_start_time
        agents_subset, heuristic = get_agents_subset_lns(
             n_neighbourhood, agents,heuristic_weights, resources_LNS1, tabu_list,nodes, nodes_dict,vc_hard_np,ec_hard_np,pc_hard_np,step_iter, h_dict
        )
        agents_outer: List[AgentAlg] = [a for a in agents if a not in agents_subset]
        planning_successful = solve_subset_with_prp_LNS1_PIBT_LNS2(
            agents_subset, agents_outer, nodes, nodes_dict, h_dict, map_dim, iter_start_time,
            pf_alg_name, pf_alg, vc_hard_np, ec_hard_np, pc_hard_np, resources_LNS1, max_path_len, k_limit, agents,0, params['pid'], tabu_list
        )
        if planning_successful:
            updated_delays, tabu_list = update_paths_if_total_shorter_LNS1(agents_subset,vc_hard_np, ec_hard_np, pc_hard_np, 0, tabu_list,params['k_limit'],h_dict)
            update_weights(heuristic, updated_delays, heuristic_weights, learning_rate=0.03)
            if updated_delays != 0:
                improved = True
        for agent in agents_subset:
            if not all_goals_reached :
                new_dist = calculate_delay(agent.k_path[-1],agent.goal_node, h_dict)
            else:
                new_dist = calculate_arrival_time_single_agent(agent)
            agent_distances[agent.num]['raw'] = new_dist
        resources_LNS1.return_resources()
    for agent in agents:
        if len(agent.k_path) == None or len(agent.k_path) <= 1:
            wait_path = [agent.curr_node] * (k_limit + 1)
            agent.k_path = wait_path[:]
    repair_agents_k_paths(agents, k_limit)
    return improved

def run_choice_alg_improve_LNS1(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict,
) -> Tuple[Dict[str, List[Node]] | None, dict]:
    """
    MAPF:
    - stop condition: all agents at their locations or time is up
    - behaviour, when agent is at its goal: the goal remains the same
    - output: success, time, makespan, soc
    -> LMAPF:
    - stop condition: the end of n iterations where every iteration has a time limit
    - behaviour, when agent is at its goal: agent receives a new goal
    - output: throughput
    """
    n_steps: int = params['n_steps']
    k_limit: int = params['k_limit']
    alg_name: bool = params['alg_name']
    to_render: bool = params['final_render']
    img_np: np.ndarray = params['img_np']
    choise_counter = [0,0]
    heuristic_counter = {'Shared': 0, 'CPB': 0, 'RCPB': 0, 'PID': 0}
    rng = np.random.RandomState(seed=params['scene_index'])
    params['rng'] = rng
    global_start_time = time.time()
    throughput: int = 0

    if to_render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(14, 7))

    agents, agents_dict = create_lns_agents(start_nodes, goal_nodes)
    vc_empty_np, ec_empty_np, pc_empty_np = init_constraints(map_dim, k_limit + 1)
    n_agents = len(agents)
    if 'PIBT' in params['distribution_name']:
        params['distribution_name'] = 'fixed-PIBT'
    path_len = 0
    excess_budget = 0
    LNS1_counter = 0
    for step_iter in range(n_steps):
        improved = False
        if step_iter == path_len:
            start_nodes = [agent.curr_node for agent in agents]
            goal_nodes = [agent.goal_node for agent in agents]
            LNS_agents, LNS_agents_dict = create_lns_agents(start_nodes, goal_nodes)
            for agent in LNS_agents:
                agent.k_path = []
            if params['distribution_name'] != 'fixed-PIBT':
                excess_budget = solve_k_LNS2(
                    LNS_agents, nodes, nodes_dict, h_dict, map_dim, vc_empty_np, ec_empty_np, pc_empty_np, params,step_iter, excess_budget,heuristic_counter
                )
            agents_paths, pibt_info, pibt_agents = run_prefix_pibt(start_nodes,goal_nodes, nodes, nodes_dict, h_dict, map_dim, params)
            turn_pibt_path_to_k_path(pibt_agents,step_iter,k_limit)
            if params['distribution_name'] != 'fixed-PIBT':
                chosen_agents = calculate_state_score(LNS_agents,pibt_agents,choise_counter,h_dict)
            else:
                chosen_agents = pibt_agents
                excess_budget = params['resources_per_agent']*params['number_of_agents']
            if excess_budget != 0 and 'LNS1' in params['alg_name']:
                improved = improve_LNS1_shared(chosen_agents,nodes,nodes_dict,h_dict,map_dim,params,step_iter,excess_budget)
                excess_budget = 0
                if improved == True:
                    LNS1_counter += 1
            add_k_paths_to_agents(chosen_agents)
            for chosen_agent in chosen_agents:
                corresponding_agent = next((a for a in agents if a.name == chosen_agent.name), None)
                if corresponding_agent is not None:
                    if len(corresponding_agent.path) == 0:
                        corresponding_agent.path.extend(chosen_agent.path)
                    else:
                        corresponding_agent.path.extend(chosen_agent.path[1:])
            path_len = len(agents[0].path)
            result = validate_solution(agents)
            if result['is_valid'] == False:
                print(result)


        for agent in agents:
            agent.curr_node = agent.path[step_iter]
        if 'Lifelong-LNS2-PIBT' in params['alg_name']:
            throughput += update_goal_nodes(agents, nodes)
        if 'k-LNS2-PIBT' in params['alg_name'] and all_agents_reached_goal(agents):
            path_len_arr = []
            for agent in agents:
                step = 0
                for path in agent.path:
                    if path != agent.goal_node:
                        step += 1
                    else:
                        path_len_arr.append(step)
            break

        global_runtime = time.time() - global_start_time

        if to_render:
            index = [i for i, agent in enumerate(agents) if agent.num == 113]
            i_agent = agents[index[0]]
            plot_info = {
                'img_np': img_np,
                'agents': agents,
                'i_agent': i_agent,
                'i': step_iter,
            }
            plot_step_in_env(ax, plot_info)
            plt.pause(0.001)
            ax.clear()
    soc = calculate_soc(agents)
    Shared_counter, CPB_counter, RCPB_counter, PID_counter = heuristic_counter['Shared'],heuristic_counter['CPB'],heuristic_counter['RCPB'],heuristic_counter['PID']
    print(f'counter: Shared:{Shared_counter} | CPB: {CPB_counter} | RCPB: {RCPB_counter} | PID: {PID_counter}')
    print(f"resources per agent: {params['resources_per_agent']} | makespen: {step_iter+1} | soc:{soc} | throughput: {throughput} |  scene:{params['scene_index']} | chosen [LNS,PIBT]: {choise_counter}")
    return {a.name: a.path for a in agents}, {'agents': agents, 'throughput': throughput, 'makespen': step_iter + 1, 'soc':soc, 'LNS2 counter':choise_counter[0],'PIBT counter':choise_counter[1], 'LNS1 counter':LNS1_counter,'Shared counter': Shared_counter,'CPB counter': CPB_counter,'RCPB counter': RCPB_counter,'PID counter': PID_counter}

def run_choice_alg_spillover(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict,
) -> Tuple[Dict[str, List[Node]] | None, dict]:
    """
    MAPF:
    - stop condition: all agents at their locations or time is up
    - behaviour, when agent is at its goal: the goal remains the same
    - output: success, time, makespan, soc
    -> LMAPF:
    - stop condition: the end of n iterations where every iteration has a time limit
    - behaviour, when agent is at its goal: agent receives a new goal
    - output: throughput
    """
    n_steps: int = params['n_steps']
    k_limit: int = params['k_limit']
    alg_name: bool = params['alg_name']
    to_render: bool = params['final_render']
    img_np: np.ndarray = params['img_np']
    choise_counter = [0,0]

    global_start_time = time.time()
    throughput: int = 0

    if to_render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(14, 7))

    agents, agents_dict = create_lns_agents(start_nodes, goal_nodes)
    vc_empty_np, ec_empty_np, pc_empty_np = init_constraints(map_dim, k_limit + 1)
    n_agents = len(agents)

    path_len = 0
    excess_budget = 0
    for step_iter in range(n_steps):
        if step_iter == path_len:
            start_nodes = [agent.curr_node for agent in agents]
            goal_nodes = [agent.goal_node for agent in agents]
            LNS_agents, LNS_agents_dict = create_lns_agents(start_nodes, goal_nodes)
            for agent in LNS_agents:
                agent.k_path = []
            excess_budget = solve_k_LNS2(
                LNS_agents, nodes, nodes_dict, h_dict, map_dim, vc_empty_np, ec_empty_np, pc_empty_np, params,step_iter, excess_budget
            )
            agents_paths, pibt_info, pibt_agents = run_prefix_pibt(start_nodes,goal_nodes, nodes, nodes_dict, h_dict, map_dim, params)
            turn_pibt_path_to_k_path(pibt_agents,step_iter,k_limit)
            chosen_agents = calculate_state_score(LNS_agents,pibt_agents,choise_counter,h_dict)
            add_k_paths_to_agents(chosen_agents)
            for chosen_agent in chosen_agents:
                corresponding_agent = next((a for a in agents if a.name == chosen_agent.name), None)
                if corresponding_agent is not None:
                    if len(corresponding_agent.path) == 0:
                        corresponding_agent.path.extend(chosen_agent.path)
                    else:
                        corresponding_agent.path.extend(chosen_agent.path[1:])
            path_len = len(agents[0].path)
            result = validate_solution(agents)
            if result['is_valid'] == False:
                print(result)


        for agent in agents:
            agent.curr_node = agent.path[step_iter]
        if 'Lifelong-LNS2-PIBT-spillover' in params['alg_name']:
            throughput += update_goal_nodes(agents, nodes)
        if 'k-LNS2-PIBT-spillover' in params['alg_name'] and all_agents_reached_goal(agents):
            path_len_arr = []
            for agent in agents:
                step = 0
                for path in agent.path:
                    if path != agent.goal_node:
                        step += 1
                    else:
                        path_len_arr.append(step)
            break

        global_runtime = time.time() - global_start_time

        if to_render:
            index = [i for i, agent in enumerate(agents) if agent.num == 113]
            i_agent = agents[index[0]]
            plot_info = {
                'img_np': img_np,
                'agents': agents,
                'i_agent': i_agent,
                'i': step_iter,
            }
            plot_step_in_env(ax, plot_info)
            plt.pause(0.001)
            ax.clear()
    soc = calculate_soc(agents)
    print(f"makespen: {step_iter + 1} | soc:{soc}chosen [LNS,PIBT]: {choise_counter}")
    return {a.name: a.path for a in agents}, {'agents': agents, 'throughput': throughput, 'makespen': step_iter + 1, 'soc':soc}



@use_profiler(save_dir='../stats/alg_lifelong_LNS2.pstat')
def run_best_of_excess_budget(map_name, number_of_agents, alg_name, distribution_function, neighbourhood_distribution, neighbourhood_agents_class, resources_per_agent, prefix,pid,folder_name, scene_index):
    to_render = False

    n_neighbourhood: int = 5
    k_limit: int = prefix
    if neighbourhood_distribution is not None:
        resources = resource_and_neib_distributions.create(distribution_function, neighbourhood_distribution,
                                                       neighbourhood_agents_class, resources_per_agent, number_of_agents)
    else:
        resources = distribution_function

    params_lifelong_lns_sipps = {
        'map': map_name,
        'number_of_agents': number_of_agents,
        'max_iter_time': 5,
        'n_steps': 200,
        'alg_name': alg_name,
        'constr_type': 'soft',
        'k_limit': k_limit,
        'n_neighbourhood': n_neighbourhood,
        'pf_alg_name': 'sipps',
        'pf_alg': run_sipps,
        'distribution_function': resources,
        'distribution_name': resources.class_name,
        'resources_per_agent': resources_per_agent,
        'max_depth_search': k_limit,
        'scene_index': scene_index,
        'final_render': to_render,
        'folder_name': folder_name,
        'pid': pid,
        'to_save': True,
    }
    if 'spillover' in alg_name:
        run_mapf_alg(alg=run_choice_alg_spillover, params=params_lifelong_lns_sipps)
    else:
        run_mapf_alg(alg=run_choice_alg_improve_LNS1, params=params_lifelong_lns_sipps)

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    number_of_agents_arrays = [3,6,7,12]
    pid = [1, 1.5, 1]
    scenes = [3,9,13,19,25]
    resources_array = [50,100,150,200]
    prefix_array = [10,15,20]
    res = 10
    prefix = 10
    run_best_of_excess_budget('4_7_soft_lock.map', 10, f'k-LNS2-PIBT-LNS1', fixed_distribution,
                              neib_shared,
                              neib_agents_shared, res, prefix, pid, "prefix-budget",1)
    run_best_of_excess_budget('4_7_soft_lock.map', 10, f'k-LNS2-PIBT-LNS1', fixed_distribution,
                              neib_proportions_with_prefix,
                              neib_agents_shared, res, prefix, pid, "prefix-budget",1)
    run_best_of_excess_budget('4_7_soft_lock.map', 10, f'k-LNS2-PIBT-LNS1', fixed_distribution,
                              neib_proportions_with_prefix_reversed,
                              neib_agents_shared, res, prefix, pid, "prefix-budget",1)
    run_best_of_excess_budget('4_7_soft_lock.map', 10, f'k-LNS2-PIBT-LNS1', fixed_distribution,
                              neib_pid,
                              neib_agents_shared, res, prefix, pid, "prefix-budget",1)
    run_best_of_excess_budget('4_7_soft_lock.map', 10, f'k-LNS2-PIBT-LNS1', fixed_distribution,
                              neib_multi_arm_bandit,
                              neib_agents_shared, res, prefix, pid, "prefix-budget",1)
