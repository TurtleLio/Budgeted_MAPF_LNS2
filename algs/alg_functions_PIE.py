import numpy as np
from globals import *
from algs.alg_functions_LNS2 import *
from algs.alg_limited_temporal_a_star_neighb import *
from valid_check import *

def two_equal_paths_have_confs_PIE(path1: List[Node], path2: List[Node], k_limit: int):
    assert len(path1) == len(path2)
    from1 = None
    from2 = None
    for i in range(k_limit):
        to1, to2 = path1[i], path2[i]
        if to1.x == to2.x and to1.y == to2.y:
            return True
        if i > 0:
            if from1.x == to2.x and from1.y == to2.y and to1.x == from2.x and to1.y == from2.y:
                return True
        from1 = to1
        from2 = to2
    return False


def get_limit_cp_graph(
        agents: List[AgentAlg],
        other_agents: List[AgentAlg] | None = None,
        prev_cp_graph: Dict[str, List[AgentAlg]] | None = None,
        k_limit: int = int(1e10)
) -> Tuple[Dict[str, List[AgentAlg]], Dict[str, List[str]]]:
    if other_agents is None:
        other_agents = []
    # align_all_paths(agents)
    cp_graph: Dict[str, List[AgentAlg]] = {}
    for a1, a2 in combinations(agents, 2):
        if exceeds_k_dist(a1.curr_node, a2.curr_node, k_limit + 1):
            continue
        if two_equal_paths_have_confs(a1.path, a2.path):
            if a1.name not in cp_graph:
                cp_graph[a1.name] = []
            if a2.name not in cp_graph:
                cp_graph[a2.name] = []
            cp_graph[a1.name].append(a2)
            cp_graph[a2.name].append(a1)
    for other_a in other_agents:
        if other_a.name in prev_cp_graph:
            if other_a.name not in cp_graph:
                cp_graph[other_a.name] = []
            for nei in prev_cp_graph[other_a.name]:
                if nei not in agents:
                    cp_graph[other_a.name].append(nei)
        for a in agents:
            if exceeds_k_dist(other_a.curr_node, a.curr_node, k_limit + 1):
                continue
            if two_equal_paths_have_confs_PIE(other_a.path, a.path, k_limit):
                if other_a.name not in cp_graph:
                    cp_graph[other_a.name] = []
                if a.name not in cp_graph:
                    cp_graph[a.name] = []
                cp_graph[other_a.name].append(a)
                cp_graph[a.name].append(other_a)
    return cp_graph, {}


def update_paths_if_total_shorter(agents, vc_hard_np, ec_hard_np, pc_hard_np, step_iter, tabu_list):
    failed_agents = [agent for agent in agents if len(agent.k_path) == 0]
    if len(failed_agents) > 0:
        # Some agents failed, automatically reject entire solution
        #print(f"❌ REJECTING k_paths: {len(failed_agents)} agents failed to find paths")
        # print(f"Failed agents: {[agent.num for agent in failed_agents]}")
        #remove_agents_constraints(agents, vc_hard_np, ec_hard_np, pc_hard_np)
        remove_agents_constraints_fast(agents, vc_hard_np, ec_hard_np, pc_hard_np,step_iter)
        #validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_removal")
        for agent in agents:
            #update_constraints(agent.temp_path, vc_hard_np, ec_hard_np, pc_hard_np, agent.num)
            extend_agent_temp_path_to_length(agent,step_iter)
            agent.path = agent.path[:step_iter] + agent.temp_path
            update_constraints_tracked(agent.path, vc_hard_np, ec_hard_np, pc_hard_np, agent)
        #validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_addition")
        tabu_list.append(agents[0])
        return 0, tabu_list
    total_path_len = 0
    total_k_path_len = 0
    #--------------------------------------------------------
    k_len_arr=[]
    temp_len_arr=[]
    for agent in agents:
        #temp_len = agent.current_path_length
        temp_len = calculate_effective_path_length(agent.temp_path, agent.goal_node)
        k_len = calculate_effective_path_length(agent.k_path, agent.goal_node)
        # total_path_len += temp_len
        # total_k_path_len += k_len
        k_len_arr.append(k_len)
        temp_len_arr.append(temp_len)
    #----------------------------------------------------------
    # total_path_len = max(temp_len_arr)
    # total_k_path_len = max(k_len_arr)
    total_path_len = sum(temp_len_arr)
    total_k_path_len = sum(k_len_arr)
        # Add debug
        #print(f"Agent {agent.num}: temp_path={temp_len}, k_path={k_len}")

    #print(f"TOTAL: temp_paths={total_path_len}, k_paths={total_k_path_len}")

    if total_k_path_len < total_path_len:
        print(f"✅ ACCEPTING k_paths | old:{temp_len_arr} | new: {k_len_arr}")
        # agent_nums_in_subset = [agent.num for agent in agents]
        # agents_47_and_231_in_subset = 47 in agent_nums_in_subset and 231 in agent_nums_in_subset
        # if agents_47_and_231_in_subset:
        #     print(f"⚠️ Agents 47 and 231 are both in current subset: {agent_nums_in_subset}")
        remove_agents_constraints_fast(agents, vc_hard_np, ec_hard_np, pc_hard_np,step_iter)
        for agent in agents:
            #debug_agent_68_paths(agent, "BEFORE_ACCEPTING_K_PATH", step_iter)
            agent.temp_path =agent.k_path
            #debug_path_reconstruction(agent, step_iter)
            extend_agent_temp_path_to_length(agent, step_iter)
            agent.path = agent.path[:step_iter] + agent.temp_path
            update_constraints_tracked(agent.path, vc_hard_np, ec_hard_np, pc_hard_np, agent)
            #agent.path = agent.path[:step_iter+1] + agent.temp_path
            agent.current_path_length = calculate_effective_path_length(agent.temp_path, agent.goal_node)
            #debug_agent_68_paths(agent, "AFTER_ACCEPTING_K_PATH", step_iter)
        return (total_path_len - total_k_path_len), tabu_list
    else:
        #print("❌ REJECTING k_paths, keeping temp_paths")
        #remove_agents_constraints(agents, vc_hard_np, ec_hard_np, pc_hard_np)
        remove_agents_constraints_fast(agents, vc_hard_np, ec_hard_np, pc_hard_np,step_iter)
        #validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_removal")
        for agent in agents:
            #agent.temp_path = agent.temp_path[1:]
            #debug_agent_68_paths(agent, "BEFORE_REJECTING_K_PATH", step_iter)
            if len(agent.temp_path) != 0:
                #update_constraints(agent.temp_path, vc_hard_np, ec_hard_np, pc_hard_np,agent.num)
                extend_agent_temp_path_to_length(agent, step_iter)
                agent.path = agent.path[:step_iter] + agent.temp_path
                update_constraints_tracked(agent.path, vc_hard_np, ec_hard_np, pc_hard_np, agent)
                #validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_addition")
            else:
                #update_constraints(agent.path[:step_iter], vc_hard_np, ec_hard_np, pc_hard_np, agent.num)
                update_constraints_tracked(agent.path, vc_hard_np, ec_hard_np, pc_hard_np, agent)
                #validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_addition")
            #debug_agent_68_paths(agent, "AFTER_REJECTING_K_PATH", step_iter)
            #debug_path_reconstruction(agent, step_iter)
            # agent.path = agent.path[:step_iter+1] + agent.temp_path
        tabu_list.append(agents[0])
        return 0, tabu_list


def select_heuristics(weights):
    total = sum(weights.values())
    prob = {h: w/total for h, w in weights.items()}
    return np.random.choice(list(prob.keys()), p=list(prob.values()))
def update_weights(heuristic, improvement, weights, learning_rate=0.01):
    if improvement > 0:
        weights[heuristic] = learning_rate*improvement + (1-learning_rate)* weights[heuristic]
    else:
        weights[heuristic] *= (1-learning_rate)

def get_agents_subset_lns(n_neighborhood,agents, heuristic_weights, resources, tabu_list,nodes, nodes_dict,vc_hard_np,ec_hard_np,pc_hard_np,step_iter, h_dict):
    heuristic = select_heuristics(heuristic_weights)
    #print(f'heuristic chosen:{heuristic} | weights:{heuristic_weights}')
    agents_subset = []
    if heuristic == 'agent_based':
        delays = calculate_delays(agents, h_dict)
        if len(agents) <= len(tabu_list) - n_neighborhood:
            while heuristic == 'agent_based':
                update_weights(heuristic, -1, heuristic_weights, 0.02)
                heuristic = select_heuristics(heuristic_weights)
        else:
            agent = max([a for a in agents if a not in tabu_list], key=lambda a: delays[a])
            random_walk_info = weighted_random_walk_distance_based_nodes(agent.curr_node, agent.goal_node, nodes,
                                                  nodes_dict, max_steps=20, exploration_rate=0.2)
            conflicting_agents = check_path_conflicts(random_walk_info['path'],vc_hard_np,ec_hard_np,pc_hard_np, agents)
            agents_subset.append(agent)
            conflicting_index = 0
            for i in range(n_neighborhood - 1):
                # Find next conflicting agent that's not already in agents_subset
                while conflicting_index < len(conflicting_agents):
                    if conflicting_agents[conflicting_index] not in agents_subset:
                        agents_subset.append(conflicting_agents[conflicting_index])
                        conflicting_index += 1
                        break
                    conflicting_index += 1
                else:
                    # No more unique conflicting agents, pick random
                    available_agents = [agent for agent in agents if agent not in agents_subset]
                    if available_agents:
                        random_agent = random.choice(available_agents)
                        agents_subset.append(random_agent)
        # for i in range(n_neighborhood-1):
        #     if i < len(conflicting_agents):
        #         agents_subset.append(conflicting_agents[i])
        #     else:
        #         available_agents = [agent for agent in agents if agent not in agents_subset]
        #         if available_agents:
        #             random_agent = random.choice(available_agents)
        #             agents_subset.append(random_agent)
    if heuristic == 'random':
        agents_subset = random.sample(agents, n_neighborhood)
    if heuristic == 'map_based':
        agents_rank = analyze_most_visited_vertices(vc_hard_np, agents, 0)
        agents_subset = agents_rank[:n_neighborhood]
    return agents_subset, heuristic


def check_path_conflicts(path: List, vc_hard_np: np.ndarray, ec_hard_np: np.ndarray,
                         pc_hard_np: np.ndarray, agents) -> List[int]:
    conflicting_agents = set()

    # Check each step in the path
    for time_step, node in enumerate(path):
        x, y = node.x, node.y
        # Check vertex constraint (VC) - another agent at same position and time
        if time_step < vc_hard_np.shape[2]:  # Within time bounds
            existing_agent = vc_hard_np[x, y, time_step]
            if existing_agent != 0:  # 0 means no agent, any other value is an agent ID
                conflicting_agents.add(int(existing_agent))
        # Check permanent constraint (PC) - another agent permanently at this position
        permanent_agent = pc_hard_np[x, y]
        if permanent_agent != -1:  # -1 means no permanent agent
            conflicting_agents.add(int(permanent_agent))
        # Check edge constraint (EC) - another agent using the same edge at same time
        if time_step > 0 and time_step < ec_hard_np.shape[4]:  # Not first step and within time bounds
            prev_node = path[time_step - 1]
            x1, y1 = prev_node.x, prev_node.y
            x2, y2 = node.x, node.y
            # Check the edge from previous position to current position
            existing_agent = ec_hard_np[x1, y1, x2, y2, time_step]
            if existing_agent != 0:  # 0 means no agent using this edge
                conflicting_agents.add(int(existing_agent))
            # Also check the reverse edge (in case of head-on collision)
            reverse_agent = ec_hard_np[x2, y2, x1, y1, time_step]
            if reverse_agent != 0:
                conflicting_agents.add(int(reverse_agent))
        agent_id_to_agent = {agent.num: agent for agent in agents}
        conflicting_agent_objects = [agent_id_to_agent[agent_id] for agent_id in conflicting_agents
                                     if agent_id in agent_id_to_agent]

    return conflicting_agent_objects

def calculate_delays(agents, h_dict):
    delays = {}
    for agent in agents:
        # Get path length (number of nodes in path)
        #path_length = calculate_effective_path_length(agent.path, agent.goal_node)
        path_length = agent.current_path_length
        #path_length = len(agent.path) if hasattr(agent, 'path') and agent.path else 0

        # Calculate Manhattan distance from current position to goal
        current_pos = agent.curr_node if hasattr(agent, 'current_node') else agent.start_node
        goal_pos = agent.goal_node

        #manhattan_dist = abs(goal_pos.x - current_pos.x) + abs(goal_pos.y - current_pos.y)
        goal_heuristic_map = h_dict[goal_pos.xy_name]
        shortest_path_distance = int(goal_heuristic_map[current_pos.x, current_pos.y])
        # Delay = path length - Manhattan distance
        delay = path_length - shortest_path_distance
        delays[agent] = delay
    return delays

def manhattan_distance_nodes(node1: Node, node2: Node) -> int:
    """Calculate Manhattan distance between two nodes."""
    return abs(node2.x - node1.x) + abs(node2.y - node1.y)


def weighted_random_walk_distance_based_nodes(start_node: Node, goal_node: Node, nodes: List[Node],
                                              nodes_dict: Dict[str, Node], max_steps: int = 20, exploration_rate: float = 0.2) -> Dict:

    current_node = start_node
    path = [current_node]

    for step in range(max_steps):
        if current_node == goal_node:
            # if verbose:
            #     print(f"Goal reached in {step} steps!")
            return {'path': path, 'steps': step, 'success': True}

        if not current_node.neighbours_nodes:
            # if verbose:
            #     print(f"No neighbors available at {current_node}")
            break

        # Calculate current distance to goal
        current_distance = manhattan_distance_nodes(current_node, goal_node)

        # Calculate weights based on distance reduction
        weights = []
        for neighbor in current_node.neighbours_nodes:
            new_distance = manhattan_distance_nodes(neighbor, goal_node)

            # Higher weight for moves that reduce distance to goal
            if new_distance < current_distance:
                weights.append(1.0 + (1.0 - exploration_rate))
            else:
                weights.append(exploration_rate)

        # Select next node based on weights
        next_node = random.choices(current_node.neighbours_nodes, weights=weights)[0]

        current_node = next_node
        path.append(current_node)

        # if verbose and step % 100 == 0:
        #     print(
        #         f"Step {step}: at {current_node}, distance to goal: {manhattan_distance_nodes(current_node, goal_node)}")

    return {'path': path, 'steps': max_steps, 'success': False}


def analyze_most_visited_vertices(vc_np: np.ndarray,agents, start_time: int = 0):
    vertex_visits = defaultdict(lambda: {'agents': set(), 'time_steps': []})
    # Get dimensions
    max_x, max_y, max_time = vc_np.shape

    # Analyze from start_time onward
    for t in range(start_time, max_time):
        for x in range(max_x):
            for y in range(max_y):
                agent_id = vc_np[x, y, t]
                if agent_id != 0:  # If there's an agent at this position
                    vertex = (x, y)
                    vertex_visits[vertex]['agents'].add(int(agent_id))
                    vertex_visits[vertex]['time_steps'].append(t)

    # Convert to list and sort by visit count
    result = []
    for vertex, data in vertex_visits.items():
        result.append({
            'vertex': vertex,
            'visit_count': len(data['time_steps']),
            'agents': sorted(list(data['agents'])),
            'time_steps': sorted(data['time_steps'])
        })

    # Sort by visit count (descending), then by vertex coordinates for tie-breaking
    result.sort(key=lambda x: (-x['visit_count'], x['vertex']))
    agent_priorities = {}  # agent_id -> highest vertex rank they appear in

    # Assign priority based on vertex rank (0 = most visited vertex)
    for rank, vertex_info in enumerate(result):
        for agent in vertex_info['agents']:
            if agent not in agent_priorities:
                agent_priorities[agent] = rank

    # Sort agents by their priority (lower rank = higher priority)
    sorted_agents_id = sorted(agent_priorities.keys(), key=lambda x: agent_priorities[x])
    # Create a mapping from agent ID to agent object
    agent_id_to_agent = {agent.num: agent for agent in agents}
    # Convert agent IDs back to agent objects
    sorted_agents = [agent_id_to_agent[agent_id] for agent_id in sorted_agents_id
                     if agent_id in agent_id_to_agent]
    return sorted_agents

# def solve_subset_with_prp_PIE(
#         agents_subset: List[AgentAlg],
#         outer_agents: List[AgentAlg],
#         nodes: List[Node],
#         nodes_dict: Dict[str, Node],
#         h_dict: Dict[str, np.ndarray],
#         map_dim: Tuple[int, int],
#         start_time: int | float,
#         pf_alg_name: str,
#         pf_alg,
#         vc_hard_np, ec_hard_np, pc_hard_np,
#         resources,
#         max_path_len,
#         k_limit: int = int(1e10),
#         agents: List[AgentAlg] | None = None,
#         step_iter: int = int(0),
#         pid: List[float] | None = None,
#         tabu_list: List[AgentAlg] | None = None,
# ) -> None:
#     c_sum: int = 0
#     h_priority_agents: List[AgentAlg] = outer_agents[:]
#
#     si_table: Dict[str, List[Tuple[int, int, str]]] = init_si_table(nodes)
#     longest_len = max([len(a.path) for a in agents])
#     vc_soft_np, ec_soft_np, pc_soft_np = init_constraints(map_dim, longest_len)
#     # vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, longest_len)
#     # #-----------------------------------------------
#     # for agent in agents:
#     #     if len(agent.temp_path) != 0:
#     #         #update_constraints(agent.temp_path, vc_hard_np, ec_hard_np, pc_hard_np, agent.num)
#     #         update_constraints_tracked(agent.temp_path, vc_hard_np, ec_hard_np, pc_hard_np, agent)
#     #         validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_addition")
#     #     else:
#     #         #update_constraints(agent.path[:step_iter], vc_hard_np, ec_hard_np, pc_hard_np, agent.num)
#     #         update_constraints_tracked(agent.path[:step_iter], vc_hard_np, ec_hard_np, pc_hard_np, agent)
#     #         validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_addition")
#     # #------------------------------------------------
#     # ec_hard_np = init_ec_table(map_dim, longest_len)
#     # ec_soft_np = init_ec_table(map_dim, longest_len)
#     # #---------------------------------------------------------
#     # for h_agent in h_priority_agents:
#     #     # update_constraints(h_agent.path, vc_soft_np, ec_soft_np, pc_soft_np)
#     #     update_ec_table(h_agent.path, ec_soft_np)
#     #     si_table = update_si_table_soft(h_agent.path, si_table)
#     # #------------------------------------------------------------
#     #remove_agents_constraints(agents_subset, vc_hard_np, ec_hard_np, pc_hard_np)
#     remove_agents_constraints_fast(agents_subset, vc_hard_np, ec_hard_np, pc_hard_np,step_iter)
#     #validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_removal")
#     #random.shuffle(agents_subset)
#     resources.agent_subset = agents_subset
#     resources.neib_pool(agents_subset,prefix = k_limit, pid = pid)
#     resources.agent_distribution()
#     planning_successful = True
#     # Check if specific agents are in the current subset
#     agent_nums_in_subset = [agent.num for agent in agents_subset]
#     agents_0_or_104_in_subset = 0 in agent_nums_in_subset or 104 in agent_nums_in_subset
#
#     # if agents_0_or_104_in_subset:
#     #     print(f"⚠️ Agents 0 or 104 are in current subset: {agent_nums_in_subset}")
#     for agent in agents_subset:
#         # new_path, a_info = run_limited_temporal_a_star(
#         #     agent.start_node, agent.goal_node, nodes, nodes_dict, h_dict,
#         #     vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, resources,
#         #     agent=agent, si_table=si_table, is_neighborhood = True
#         # )
#         # if agent.num == 0:
#         #     print("stop")
#         #validate_constraint_arrays(vc_hard_np, ec_hard_np, pc_hard_np)
#
#         # new_path, a_info = run_limited_temporal_a_star(
#         #     agent.curr_node, agent.goal_node, nodes, nodes_dict, h_dict,
#         #     vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, resources,
#         #     agent=agent, si_table=si_table, is_neighborhood = True
#         # )
#         # print(f"🕐 Initializing A*...")
#         #debug_agent_68_paths(agent, "BEFORE_REPLANNING", step_iter)
#         A_time = time.time()
#         if agent.num == 0:
#             print("stop")
#         new_path, a_info = run_limited_temporal_a_star_optimized(
#             agent.curr_node, agent.goal_node, nodes, nodes_dict, h_dict,
#             vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, resources, max_depth=max_path_len,
#             agent=agent, si_table=si_table, is_neighborhood = True, step_iter=step_iter,
#         )
#         #debug_agent_68_paths(agent, "AFTER_A_STAR", step_iter, new_path)
#         # print(f"✅ A* done in {time.time() - A_time:.2f}s")
#         # if new_path is None:
#         #     agent.k_path = None
#         #     break
#         if new_path == None:
#             #print(f"❌ Agent {agent.num} failed, aborting planning for all agents")
#             planning_successful = False
#             break
#         #print(f"found a path for agent: {agent.num} | resources left:{sum(resources.max_nodes)}")
#         agent.k_path = new_path
#         #debug_agent_68_paths(agent, "AFTER_K_PATH_SET", step_iter)
#         k_path_all = agent.path[:step_iter]+agent.k_path
#         update_constraints_tracked(k_path_all, vc_hard_np, ec_hard_np, pc_hard_np, agent)
#         #validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_addition")
#         h_priority_agents.append(agent)
#         #align_all_paths(h_priority_agents)
#
#         #c_sum += sipps_info['c']
#     if not planning_successful:
#         # ✅ Handle failure case - revert to temp_paths immediately
#         #print("Planning failed, reverting all agents to temp_paths")
#         #remove_agents_constraints(agents_subset, vc_hard_np, ec_hard_np, pc_hard_np)
#         remove_agents_constraints_fast(agents_subset, vc_hard_np, ec_hard_np, pc_hard_np,step_iter)
#         #validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_removal")
#         for agent in agents_subset:
#             #update_constraints(agent.temp_path, vc_hard_np, ec_hard_np, pc_hard_np, agent.num)
#             agent.path = agent.path[:step_iter]+agent.temp_path
#             update_constraints_tracked(agent.path, vc_hard_np, ec_hard_np, pc_hard_np, agent)
#             #validate_constraints_consistency(agents, vc_hard_np, ec_hard_np, pc_hard_np, "after_addition")
#             #agent.path = agent.path[:step_iter] + agent.temp_path
#         tabu_list.append(agents_subset[0])
#         return  # ✅ Exit early, don't call update_paths_if_total_shorter
#
#         # ✅ Only call this if ALL agents succeeded
#     # improvement, tabu_list = update_paths_if_total_shorter(
#     #     agents_subset, vc_hard_np, ec_hard_np, pc_hard_np, step_iter, tabu_list
#     # )
#     #si_table = update_si_table_soft(new_path, si_table)
#     # update_constraints(new_path, vc_hard_np, ec_hard_np, pc_hard_np, agent.num)
#     # if longest_len < len(new_path):
#     #     longest_len = len(new_path)
#     #     # vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, longest_len)
#     #     # vc_soft_np, ec_soft_np, pc_soft_np = init_constraints(map_dim, longest_len)
#     #     ec_hard_np = init_ec_table(map_dim, longest_len)
#     #     ec_soft_np = init_ec_table(map_dim, longest_len)
#     #     for h_agent in h_priority_agents:
#     #         # update_constraints(h_agent.path, vc_soft_np, ec_soft_np, pc_soft_np)
#     #         update_ec_table(h_agent.path, ec_soft_np)
#     # else:
#     #     # update_constraints(new_path, vc_soft_np, ec_soft_np, pc_soft_np)
#     #     update_ec_table(new_path, ec_soft_np)
#     #update_ec_table(new_path, ec_soft_np)
#
#     # checks
#     runtime = time.time() - start_time
#     # assert len(agents_subset) + len(outer_agents) == len(agents)
#     # print(
#     #     f'\r[nei calc] | agents: {len(h_priority_agents): <3} / {len(agents_subset) + len(outer_agents)} | {runtime= : .2f} s.',
#     #     end='')  # , end=''
#     # collisions: int = 0
#     # align_all_paths(h_priority_agents)
#     # for i in range(len(h_priority_agents[0].path)):
#     #     to_count = False if constr_type == 'hard' else True
#     #     collisions += check_vc_ec_neic_iter(h_priority_agents, i, to_count)
#     # if c_sum > 0:
#     #     print(f'{c_sum=}')
def solve_subset_with_prp_PIE(
        agents_subset: List[AgentAlg],
        outer_agents: List[AgentAlg],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        start_time: int | float,
        pf_alg_name: str,
        pf_alg,
        vc_hard_np, ec_hard_np, pc_hard_np,
        resources,
        max_path_len,
        k_limit: int = int(1e10),
        agents: List[AgentAlg] | None = None,
        step_iter: int = int(0),
        pid: List[float] | None = None,
        tabu_list: List[AgentAlg] | None = None,
) -> None:
    c_sum: int = 0
    h_priority_agents: List[AgentAlg] = outer_agents[:]

    si_table: Dict[str, List[Tuple[int, int, str]]] = init_si_table(nodes)
    longest_len = max([len(a.path) for a in agents])
    vc_soft_np, ec_soft_np, pc_soft_np = init_constraints_pie(map_dim, longest_len)
    remove_agents_constraints_fast(agents_subset, vc_hard_np, ec_hard_np, pc_hard_np,step_iter)
    resources.agent_subset = agents_subset
    resources.neib_pool(agents_subset,prefix = k_limit, pid = pid)
    resources.agent_distribution()
    planning_successful = True

    for agent in agents_subset:
        A_time = time.time()
        # if agent.num == 0:
        #     print("stop")
        new_path, a_info = run_limited_temporal_a_star_optimized(
            agent.curr_node, agent.goal_node, nodes, nodes_dict, h_dict,
            vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, resources, max_depth=max_path_len,
            agent=agent, si_table=si_table, is_neighborhood = True, step_iter=step_iter,
        )
        if new_path == None:
            planning_successful = False
            break
        agent.k_path = new_path
        k_path_all = agent.path[:step_iter]+agent.k_path
        update_constraints_tracked(k_path_all, vc_hard_np, ec_hard_np, pc_hard_np, agent, step_iter)

        h_priority_agents.append(agent)

    if not planning_successful:
        remove_agents_constraints_fast(agents_subset, vc_hard_np, ec_hard_np, pc_hard_np,step_iter)
        for agent in agents_subset:
            agent.path = agent.path[:step_iter]+agent.temp_path
            update_constraints_tracked(agent.path, vc_hard_np, ec_hard_np, pc_hard_np, agent)
        tabu_list.append(agents_subset[0])
        return  planning_successful
    runtime = time.time() - start_time
    return planning_successful


# def remove_agents_constraints(agents_subset: List, vc_np: np.ndarray,
# #                               ec_np: np.ndarray, pc_np: np.ndarray):
# #     for agent in agents_subset:
# #         path = agent.path
# #         if not path:
# #             continue  # No path to remove constraints for
# #         agent_id = agent.num
# #         # Remove permanent constraint (pc) - set back to -1
# #         last_node = path[-1]
# #         if pc_np[last_node.x, last_node.y] == len(path) - 1:
# #             pc_np[last_node.x, last_node.y] = -1
# #         # Remove vertex and edge constraints
# #         prev_n = path[0]
# #         for t, n in enumerate(path):
# #             # Remove vertex constraint only if it belongs to this agent
# #             if t < vc_np.shape[2] and vc_np[n.x, n.y, t] == agent_id:
# #                 vc_np[n.x, n.y, t] = 0
# #             # Remove edge constraint only if it belongs to this agent
# #             if t < ec_np.shape[4] and ec_np[prev_n.x, prev_n.y, n.x, n.y, t] == agent_id:
# #                 ec_np[prev_n.x, prev_n.y, n.x, n.y, t] = 0
# #             prev_n = n
def remove_agents_constraints(agents_subset: List, vc_np: np.ndarray,
                                    ec_np: np.ndarray, pc_np: np.ndarray):
    agent_ids = {agent.num for agent in agents_subset}
    # Remove vertex constraints
    for agent_id in agent_ids:
        vc_np[vc_np == agent_id] = 0
    # Remove edge constraints
    for agent_id in agent_ids:
        ec_np[ec_np == agent_id] = 0
    for agent in agents_subset:
        agent_id = agent.num
        if hasattr(agent, 'path') and agent.path:
            final_pos = agent.path[-1]
            final_time = len(agent.path) - 1
            # Only remove if the time matches (indicates this agent set it)
            if pc_np[final_pos.x, final_pos.y] == final_time:
                pc_np[final_pos.x, final_pos.y] = -1
        if hasattr(agent, 'temp_path') and agent.temp_path:
            final_pos = agent.temp_path[-1]
            final_time = len(agent.temp_path) - 1
            if pc_np[final_pos.x, final_pos.y] == final_time:
                pc_np[final_pos.x, final_pos.y] = -1
        if hasattr(agent, 'k_path') and agent.k_path:
            final_pos = agent.k_path[-1]
            final_time = len(agent.k_path) - 1
            if pc_np[final_pos.x, final_pos.y] == final_time:
                pc_np[final_pos.x, final_pos.y] = -1


def calculate_effective_path_length(path, goal_node):
    if not path:
        return 0
    # Find the first time the agent reaches the goal
    for i, node in enumerate(path):
        if node == goal_node:  # Using Node's __eq__ method
            return i + 1  # +1 because we want length, not index
    # If goal never reached, return full path length
    return len(path)


def normalize_all_paths_to_max_length(agents):
    if not agents:
        return 0
    # Find the maximum path length
    max_length = 0
    for agent in agents:
        if agent.path and len(agent.path) > max_length:
            max_length = len(agent.path)
    # Extend all paths to max_length
    for agent in agents:
        if agent.path:
            extend_path_to_length(agent.path, max_length)
    return max_length


def extend_path_to_length(path, target_length):
    if not path:
        return
    current_length = len(path)
    if current_length >= target_length:
        return  # Path is already long enough

    # Get the final node
    final_node = path[-1]

    # Add the final node repeatedly until we reach target length
    nodes_to_add = target_length - current_length
    for _ in range(nodes_to_add):
        path.append(final_node)


def update_constraints_tracked(
        path: List[Node], vc_np: np.ndarray, ec_np: np.ndarray, pc_np: np.ndarray, agent,
        start_time: int = 0  # Add start_time parameter
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Update constraints starting from start_time
    """
    # if isinstance(agent, int):
    #     print(f"⚠️  WARNING: update_constraints called with agent_id {agent} instead of agent object!")
    #     import traceback
    #     traceback.print_stack()
    #     return vc_np, ec_np, pc_np

    agent_id = agent.num

    # Only clear constraints for times >= start_time
    if start_time == 0:
        # Full path update - clear all
        agent.my_vertex_constraints.clear()
        agent.my_edge_constraints.clear()
        agent.my_permanent_constraint = None
    else:
        # Partial update - keep constraints before start_time
        agent.my_vertex_constraints = [(x, y, t) for x, y, t in agent.my_vertex_constraints if t < start_time]
        agent.my_edge_constraints = [(x1, y1, x2, y2, t) for x1, y1, x2, y2, t in agent.my_edge_constraints if
                                     t < start_time]

    # Set permanent constraint
    if len(path) > 0:
        last_node = path[-1]
        if last_node.x == agent.goal_node.x and last_node.y == agent.goal_node.y :
            last_time = len(path) - 1
            pc_np[last_node.x, last_node.y] = max(int(pc_np[last_node.x, last_node.y]), last_time)
            agent.my_permanent_constraint = (last_node.x, last_node.y)

    # Set vertex and edge constraints starting from start_time
    for t in range(len(path)):
        n = path[t]
        constraint_time = t + start_time

        # Vertex constraint
        if constraint_time < vc_np.shape[2]:
            # Check for conflicts before setting
            existing = vc_np[n.x, n.y, constraint_time]
            if existing != -1 and existing != agent_id:
                print(f"⚠️  WARNING: Overwriting constraint at ({n.x},{n.y}) time {constraint_time}: "
                      f"agent {existing} -> agent {agent_id}")

            vc_np[n.x, n.y, constraint_time] = int(agent_id)
            agent.my_vertex_constraints.append((n.x, n.y, constraint_time))

        # Edge constraint
        if t > 0:
            prev_n = path[t - 1]
            if constraint_time < ec_np.shape[4]:
                ec_np[prev_n.x, prev_n.y, n.x, n.y, constraint_time] = int(agent_id)
                agent.my_edge_constraints.append((prev_n.x, prev_n.y, n.x, n.y, constraint_time))

    return vc_np, ec_np, pc_np

def extend_agent_temp_path_to_length(agent, step_iter):
    target_length = len(agent.path) - step_iter
    if not agent.temp_path:
        return
    current_length = len(agent.temp_path)
    if current_length >= target_length:
        return  # Path is already long enough
    # Get the final node (goal position)
    final_node = agent.temp_path[-1]

    # Add the final node repeatedly until we reach target length
    nodes_to_add = target_length - current_length
    for _ in range(nodes_to_add):
        agent.temp_path.append(final_node)