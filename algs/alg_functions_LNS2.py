import random

from globals import *
from functions_general import *
from functions_plotting import *
from algs.alg_sipps import run_sipps
from algs.alg_sipps_functions import init_si_table, update_si_table_soft
from algs.alg_functions_PIE import *




def create_lns_agents(
        start_nodes: List[Node], goal_nodes: List[Node]
) -> Tuple[List[AgentAlg], Dict[str, AgentAlg]]:
    agents: List[AgentAlg] = []
    agents_dict: Dict[str, AgentAlg] = {}
    for num, (s_node, g_node) in enumerate(zip(start_nodes, goal_nodes)):
        new_agent = AgentAlg(num, s_node, g_node)
        agents.append(new_agent)
        agents_dict[new_agent.name] = new_agent
    return agents, agents_dict


def solution_is_found(agents: List[AgentAlg]):
    for agent in agents:
        if agent.path is None:
            return False
        if len(agent.path) == 0:
            return False
        if agent.path[-1] != agent.goal_node:
            return False
    return True


def get_shuffled_agents(agents: List[AgentAlg]) -> List[AgentAlg]:
    agents_copy = agents[:]
    random.shuffle(agents_copy)
    unfinished: List[AgentAlg] = [a for a in agents_copy if len(a.path) == 0 or a.path[-1] != a.goal_node]
    finished: List[AgentAlg] = [a for a in agents_copy if len(a.path) > 0 and a.path[-1] == a.goal_node]
    return [*unfinished, *finished]


def create_init_solution(
        agents: List[AgentAlg],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        constr_type: str,
        start_time: int | float,
        params: dict
):
    alg_name: str = params['alg_name']
    c_sum: int = 0
    h_priority_agents: List[AgentAlg] = []
    longest_len = 1
    ec_hard_np = init_ec_table(map_dim, longest_len)
    ec_soft_np = init_ec_table(map_dim, longest_len)
    si_table: Dict[str, List[Tuple[int, int, str]]] = init_si_table(nodes)

    for agent in agents:
        new_path, sipps_info = run_sipps(
            agent.start_node, agent.goal_node, nodes, nodes_dict, h_dict,
            None, ec_hard_np, None, None, ec_soft_np, None,
            agent=agent, si_table=si_table
        )
        if new_path is None:
            agent.path = None
            break
        agent.path = new_path[:]
        h_priority_agents.append(agent)
        align_all_paths(h_priority_agents)

        c_sum += sipps_info['c']

        si_table = update_si_table_soft(new_path, si_table)
        if longest_len < len(new_path):
            longest_len = len(new_path)
            ec_hard_np = init_ec_table(map_dim, longest_len)
            ec_soft_np = init_ec_table(map_dim, longest_len)
            for h_agent in h_priority_agents:
                update_ec_table(h_agent.path, ec_soft_np)
        else:
            update_ec_table(new_path, ec_soft_np)

        runtime = time.time() - start_time
        print(f'\r[{alg_name} - init] | agents: {len(h_priority_agents): <3} / {len(agents)} | {runtime= : .2f} s.',
              end='')


def solve_subset_with_prp(
        agents_subset: List[AgentAlg],
        outer_agents: List[AgentAlg],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        start_time: int | float,
        constr_type: str = 'soft',
        agents: List[AgentAlg] | None = None
) -> None:
    c_sum: int = 0
    h_priority_agents: List[AgentAlg] = outer_agents[:]

    si_table: Dict[str, List[Tuple[int, int, str]]] = init_si_table(nodes)
    longest_len = max([len(a.path) for a in agents])
    ec_hard_np = init_ec_table(map_dim, longest_len)
    ec_soft_np = init_ec_table(map_dim, longest_len)
    for h_agent in h_priority_agents:
        update_ec_table(h_agent.path, ec_soft_np)
        si_table = update_si_table_soft(h_agent.path, si_table)

    random.shuffle(agents_subset)
    for agent in agents_subset:
        new_path, sipps_info = run_sipps(
            agent.start_node, agent.goal_node, nodes, nodes_dict, h_dict,
            None, ec_hard_np, None, None, ec_soft_np, None,
            agent=agent, si_table=si_table
        )
        if new_path is None:
            agent.path = None
            break
        agent.path = new_path[:]
        h_priority_agents.append(agent)
        align_all_paths(h_priority_agents)

        c_sum += sipps_info['c']

        si_table = update_si_table_soft(new_path, si_table)

        update_ec_table(new_path, ec_soft_np)

        runtime = time.time() - start_time
        assert len(agents_subset) + len(outer_agents) == len(agents)
        print(
            f'\r[nei calc] | agents: {len(h_priority_agents): <3} / {len(agents_subset) + len(outer_agents)} | {runtime= : .2f} s.',
            end='')


def get_cp_graph(
        agents: List[AgentAlg],
        other_agents: List[AgentAlg] | None = None,
        prev_cp_graph: Dict[str, List[AgentAlg]] | None = None,
) -> Tuple[Dict[str, List[AgentAlg]], Dict[str, List[str]]]:
    if other_agents is None:
        other_agents = []
    cp_graph: Dict[str, List[AgentAlg]] = {}
    for a1, a2 in combinations(agents, 2):
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
            if not two_plans_have_no_confs(other_a.path, a.path):
                if other_a.name not in cp_graph:
                    cp_graph[other_a.name] = []
                if a.name not in cp_graph:
                    cp_graph[a.name] = []
                cp_graph[other_a.name].append(a)
                cp_graph[a.name].append(other_a)
    return cp_graph, {}


def get_outer_agent_via_random_walk(
        rand_agent: AgentAlg,
        agents_s: List[AgentAlg],
        occupied_from: Dict[str, AgentAlg]
) -> AgentAlg:
    next_node: Node = random.choice(rand_agent.path)
    while True:
        if next_node.xy_name in occupied_from and occupied_from[next_node.xy_name] not in agents_s:
            return occupied_from[next_node.xy_name]
        next_node = random.choice(next_node.neighbours_nodes)


def get_agent_s_from_random_walk(
        curr_agent: AgentAlg,
        cp_graph: Dict[str, List[AgentAlg]],
        n_neighbourhood: int,
) -> List[AgentAlg]:
    out_list: List[AgentAlg] = []
    next_nei: AgentAlg = curr_agent
    while len(out_list) < n_neighbourhood:
        nei_agents = cp_graph[next_nei.name]
        next_nei = random.choice(nei_agents)
        if next_nei not in out_list and random.random() < 0.7:
            out_list.append(next_nei)
    return out_list


def get_agents_subset(
        cp_graph: Dict[str, List[AgentAlg]],
        cp_graph_names: Dict[str, List[str]],
        n_neighbourhood: int,
        agents: List[AgentAlg],
        occupied_from: Dict[str, AgentAlg],
        h_dict: Dict[str, np.ndarray],
) -> List[AgentAlg]:
    agents_with_cp: List[AgentAlg] = [a for a in agents if a.name in cp_graph]
    curr_agent: AgentAlg = random.choice(agents_with_cp)

    lcc: List[AgentAlg] = []
    l_open = deque([curr_agent])
    i = 0
    while len(l_open) > 0:
        i += 1
        next_a = l_open.pop()
        heapq.heappush(lcc, next_a)
        random.shuffle(cp_graph[next_a.name])
        for nei_a in cp_graph[next_a.name]:
            if nei_a not in lcc and nei_a not in l_open:
                l_open.append(nei_a)

    agents_s: List[AgentAlg] = []
    if len(lcc) <= n_neighbourhood:
        agents_s.extend(lcc)
        while len(agents_s) < n_neighbourhood:
            rand_agent = random.choice(agents_s)
            outer_agent = get_outer_agent_via_random_walk(rand_agent, agents_s, occupied_from)
            agents_s.append(outer_agent)
        return agents_s
    else:
        agents_s = get_agent_s_from_random_walk(curr_agent, cp_graph, n_neighbourhood)
        return agents_s


def create_k_limit_init_solution(
        agents: List[AgentAlg],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        pf_alg_name: str,
        pf_alg,
        k_limit: int,
        start_time: int | float,
        vc_empty_np, ec_empty_np, pc_empty_np,
        resources,
        max_depth: int,
        params,
):
    max_time: bool = params['max_iter_time']
    h_priority_agents: List[AgentAlg] = []
    si_table: Dict[str, List[Tuple[int, int, str]]] = init_si_table(nodes)
    if pf_alg_name == 'sipps':
        vc_hard_np, ec_hard_np, pc_hard_np = vc_empty_np, ec_empty_np, pc_empty_np
        vc_soft_np, ec_soft_np, pc_soft_np = init_constraints(map_dim, k_limit + 1)
    elif pf_alg_name == 'a_star':
        vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, k_limit + 1)
        vc_soft_np, ec_soft_np, pc_soft_np = vc_empty_np, ec_empty_np, pc_empty_np
    else:
        raise RuntimeError('nono')

    for agent in agents:
        new_path, alg_info = pf_alg(
            agent.curr_node, agent.goal_node, nodes, nodes_dict, h_dict,
            vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, resources,
            max_depth, flag_k_limit=True, k_limit=k_limit, agent=agent, si_table=si_table
        )
        if new_path is None:
            new_path = [agent.curr_node]
        new_path = align_path(new_path, k_limit + 1)
        agent.k_path = new_path[:]
        h_priority_agents.append(agent)

        if pf_alg_name == 'sipps':
            update_constraints(new_path, vc_soft_np, ec_soft_np, pc_soft_np, agent.num)
            si_table = update_si_table_soft(new_path, si_table, consider_pc=False)
        elif pf_alg_name == 'a_star':
            update_constraints(new_path, vc_hard_np, ec_hard_np, pc_hard_np, agent.num)
        else:
            raise RuntimeError('nono')


def get_k_limit_cp_graph(
        agents: List[AgentAlg],
        other_agents: List[AgentAlg] | None = None,
        prev_cp_graph: Dict[str, List[AgentAlg]] | None = None,
        k_limit: int = int(1e10)
) -> Tuple[Dict[str, List[AgentAlg]], Dict[str, List[str]]]:
    if other_agents is None:
        other_agents = []
    cp_graph: Dict[str, List[AgentAlg]] = {}
    for a1, a2 in combinations(agents, 2):
        if exceeds_k_dist(a1.curr_node, a2.curr_node, 2*(k_limit+1)):
            continue
        if two_equal_paths_have_confs(a1.k_path, a2.k_path):
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
            if exceeds_k_dist(other_a.curr_node, a.curr_node, 2*(k_limit+1)):
                continue
            if two_equal_paths_have_confs(other_a.k_path, a.k_path):
                if other_a.name not in cp_graph:
                    cp_graph[other_a.name] = []
                if a.name not in cp_graph:
                    cp_graph[a.name] = []
                cp_graph[other_a.name].append(a)
                cp_graph[a.name].append(other_a)
    return {k: v for k, v in cp_graph.items() if v}, {}

def get_k_limit_outer_agent_via_random_walk(
        rand_agent: AgentAlg,
        agents_s: List[AgentAlg],
        occupied_from: Dict[str, AgentAlg]
) -> AgentAlg:
    next_node: Node = random.choice(rand_agent.k_path)
    while True:
        if next_node.xy_name in occupied_from and occupied_from[next_node.xy_name] not in agents_s and random.random() < 0.7:
            return occupied_from[next_node.xy_name]
        next_node = random.choice(next_node.neighbours_nodes)


def get_k_limit_agents_subset(
        cp_graph: Dict[str, List[AgentAlg]],
        cp_graph_names: Dict[str, List[str]],
        n_neighbourhood: int,
        agents: List[AgentAlg],
        occupied_from: Dict[str, AgentAlg],
        h_dict: Dict[str, np.ndarray],
        resources,
) -> List[AgentAlg]:
    agents_with_cp: List[AgentAlg] = [a for a in agents if a.name in cp_graph]


    curr_agent: AgentAlg = random.choice(agents_with_cp)
    lcc: List[AgentAlg] = []
    l_open = deque([curr_agent])
    while l_open:
        next_a = l_open.pop()
        heapq.heappush(lcc, next_a)
        random.shuffle(cp_graph[next_a.name])
        for nei_a in cp_graph[next_a.name]:
            if nei_a not in lcc and nei_a not in l_open:
                l_open.append(nei_a)


    agents_s: List[AgentAlg] = []

    if len(lcc) < n_neighbourhood:
        agents_s.extend(lcc)
        while len(agents_s) < n_neighbourhood:
            rand_agent = random.choice(agents_s)
            outer_agent = get_k_limit_outer_agent_via_random_walk(rand_agent, agents_s, occupied_from)
            agents_s.append(outer_agent)
        return agents_s
    else:
        agents_s = lcc[:n_neighbourhood]
        return agents_s

def get_k_limit_agents_subset_tests_only(
        cp_graph: Dict[str, List[AgentAlg]],
        cp_graph_names: Dict[str, List[str]],
        n_neighbourhood: int,
        agents: List[AgentAlg],
        occupied_from: Dict[str, AgentAlg],
        h_dict: Dict[str, np.ndarray],
        resources,
        subset_index: int,
) -> List[AgentAlg]:
    agents_sorted = sorted(agents, key=lambda a: a.num)

    start = subset_index * 2
    end = start + 2

    return agents_sorted[start:end]


def solve_k_limit_subset_with_prp(
        agents_subset: List[AgentAlg],
        outer_agents: List[AgentAlg],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        start_time: int | float,
        pf_alg_name: str,
        pf_alg,
        vc_empty_np, ec_empty_np, pc_empty_np,
        resources,
        max_depth_search,
        k_limit: int = int(1e10),
        agents: List[AgentAlg] | None = None,
        cp_graph: Dict[str, List[AgentAlg]] | None = None,
        pid: List[float] | None = None,
        distribution_name: str | None = None,
        heuristic_counter:Dict[str,int] | None = None,
        rng: np.random.RandomState | None = None,
) -> None:
    success = True
    h_priority_agents: List[AgentAlg] = outer_agents[:]
    si_table: Dict[str, List[Tuple[int, int, str]]] = init_si_table(nodes)
    if pf_alg_name == 'sipps':
        vc_hard_np, ec_hard_np, pc_hard_np = vc_empty_np.copy() , ec_empty_np.copy(), pc_empty_np.copy()
        vc_soft_np, pc_soft_np = vc_empty_np.copy(), pc_empty_np.copy()
        ec_soft_np = init_ec_table(map_dim, k_limit + 1)
        for h_agent in h_priority_agents:
            update_ec_table(h_agent.k_path, ec_soft_np)
            si_table = update_si_table_soft(h_agent.k_path, si_table, consider_pc=False)
    elif pf_alg_name == 'a_star':
        vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, k_limit + 1)
        vc_soft_np, ec_soft_np, pc_soft_np = vc_empty_np, ec_empty_np, pc_empty_np
        for h_agent in h_priority_agents:
            update_constraints(h_agent.k_path, vc_hard_np, ec_hard_np, pc_hard_np)
    else:
        raise RuntimeError('nono')

    random.shuffle(agents_subset)
    resources.agent_subset = agents_subset
    resources_before_search = resources.max_nodes[-1].copy()
    if 'Multi' in distribution_name:
        arm_resources, heuristic = multi_arm_resources(resources, resources.weights,heuristic_counter,rng)
        arm_resources.neib_pool(agents_subset, cp_graph = cp_graph,prefix = k_limit, pid = pid)
        resources.neib_budget = arm_resources.neib_budget
        resources.proportions = arm_resources.proportions.copy()
        neib_budget = resources.neib_budget
        resources.max_nodes[-1] -= neib_budget
    else:
        resources.neib_pool(agents_subset, cp_graph = cp_graph,prefix = k_limit, pid = pid)
    resources.agent_distribution()
    for agent in agents_subset:
        new_path, sipps_info = pf_alg(
            agent.curr_node, agent.goal_node, nodes, nodes_dict, h_dict,
            vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, resources, max_depth_search,
            flag_k_limit=True, k_limit=k_limit, agent=agent, si_table=si_table, is_neighborhood=True,
        )
        if is_wait_path(new_path):
            success = False
        new_path = align_path(new_path, k_limit + 1)
        agent.k_path = new_path[:]
        h_priority_agents.append(agent)
        agent.k_path_delay = calculate_delay(agent.k_path[-1], agent.goal_node,h_dict)

        if pf_alg_name == 'sipps':
            update_ec_table(new_path, ec_soft_np)
            si_table = update_si_table_soft(new_path, si_table, consider_pc=False)
        elif pf_alg_name == 'a_star':
            update_constraints(new_path, vc_hard_np, ec_hard_np, pc_hard_np)
        else:
            raise RuntimeError('nono')
    resources_before_return = resources.max_nodes[-1].copy()
    resources_returned = resources.return_resources()
    if 'Multi' in distribution_name:
        remaining_resources_in_neib = resources_returned
        resources.weights[heuristic]['expansions'] += neib_budget - remaining_resources_in_neib
        if success == True:
            resources.weights[heuristic]['successes'] += 1
        update_weights_multi_arm(heuristic,success,resources.weights)


def has_resources_in_cp_graph(cp_graph: Dict[str, List[AgentAlg]], resources,
                              agents: List[AgentAlg]) -> bool:
    if resources.max_nodes[-1] > 0:
        return True
    return False


def calculate_delay(curr_node, goal_node, h_dict):
    current_pos = curr_node
    goal_pos = goal_node
    goal_heuristic_map = h_dict[goal_pos.xy_name]
    delay = int(goal_heuristic_map[current_pos.x, current_pos.y])
    return delay


def normalize_agent_distances(agent_distances: dict):
    max_raw = max(agent['raw'] for agent in agent_distances.values())
    if max_raw == 0:
        for agent in agent_distances.values():
            agent['normalized'] = 0.0
    else:
        for agent in agent_distances.values():
            agent['normalized'] = agent['raw'] / max_raw

def solve_subset_with_prp_LNS1(
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
    resources.neib_pool(agents_subset,prefix = k_limit, pid = [100,150,100])
    resources.agent_distribution()
    planning_successful = True

    for agent in agents_subset:
        A_time = time.time()
        new_path, a_info = run_limited_temporal_a_star_optimized_LNS1(
            agent.curr_node, agent.goal_node, nodes, nodes_dict, h_dict,
            vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, resources, max_depth=200,
            agent=agent, si_table=si_table, is_neighborhood = True, step_iter=step_iter,
        )
        if new_path == None:
            planning_successful = False
            break
        agent.temp_path = new_path
        k_path_all = agent.path+agent.temp_path[1:]
        update_constraints_tracked(k_path_all[step_iter:step_iter+k_limit+1], vc_hard_np, ec_hard_np, pc_hard_np, agent, step_iter)
        h_priority_agents.append(agent)

    if not planning_successful:
        remove_agents_constraints_fast(agents_subset, vc_hard_np, ec_hard_np, pc_hard_np,0)
        for agent in agents_subset:
            update_constraints_tracked(agent.path + agent.k_path[1:], vc_hard_np, ec_hard_np, pc_hard_np, agent)
        tabu_list.append(agents_subset[0])
        return  planning_successful
    runtime = time.time() - start_time
    return planning_successful

def solve_subset_with_prp_LNS1_PIBT_LNS2(
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
    resources.neib_pool(agents_subset,prefix = k_limit, pid = [1.5,1.75,1])
    resources.agent_distribution()
    planning_successful = True

    for agent in agents_subset:
        A_time = time.time()
        new_path, a_info = run_limited_temporal_a_star_optimized_LNS1(
            agent.k_path[0], agent.goal_node, nodes, nodes_dict, h_dict,
            vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, resources, max_depth=200,
            agent=agent, si_table=si_table, is_neighborhood = True, step_iter=step_iter,
        )
        if new_path == None:
            planning_successful = False
            break
        agent.temp_path = new_path
        k_path_all = agent.path+agent.temp_path[1:]
        update_constraints_tracked(k_path_all[:k_limit+1], vc_hard_np, ec_hard_np, pc_hard_np, agent, step_iter)
        h_priority_agents.append(agent)

    if not planning_successful:
        remove_agents_constraints_fast(agents_subset, vc_hard_np, ec_hard_np, pc_hard_np,0)
        for agent in agents_subset:
            update_constraints_tracked(agent.path + agent.k_path[1:], vc_hard_np, ec_hard_np, pc_hard_np, agent)
        tabu_list.append(agents_subset[0])
        return  planning_successful
    runtime = time.time() - start_time
    return planning_successful

def update_paths_if_total_shorter_LNS1(agents, vc_hard_np, ec_hard_np, pc_hard_np, step_iter, tabu_list,k_limit,h_dict):
    failed_agents = [agent for agent in agents if len(agent.temp_path) == 0]
    if len(failed_agents) > 0:
        remove_agents_constraints_fast(agents, vc_hard_np, ec_hard_np, pc_hard_np,step_iter)
        for agent in agents:
            update_constraints_tracked(agent.path+agent.k_path[1:k_limit+1], vc_hard_np, ec_hard_np, pc_hard_np, agent)
        tabu_list.append(agents[0])
        return 0, tabu_list
    total_path_len = 0
    total_k_path_len = 0
    k_len_arr=[]
    temp_len_arr=[]
    for agent in agents:
        if len(agent.temp_path) >= len(agent.k_path):
            temp_len = compute_distance(agent.temp_path[len(agent.k_path)-1], agent.goal_node,h_dict)
        else:
            temp_len = compute_distance(agent.temp_path[-1], agent.goal_node, h_dict)
        k_len = compute_distance(agent.k_path[-1], agent.goal_node,h_dict)
        k_len_arr.append(k_len)
        temp_len_arr.append(temp_len)
    total_path_len = max(temp_len_arr)
    total_k_path_len = max(k_len_arr)
    if total_path_len == 0 and total_k_path_len == 0:
        total_path_len, total_k_path_len = calculate_arrival_time(agents)
    if total_k_path_len > total_path_len:
        remove_agents_constraints_fast(agents, vc_hard_np, ec_hard_np, pc_hard_np,0)
        for agent in agents:
            agent.k_path =agent.temp_path[:k_limit+1]
            update_constraints_tracked(agent.path+agent.k_path[1:k_limit+1], vc_hard_np, ec_hard_np, pc_hard_np, agent)
        return (total_path_len - total_k_path_len), tabu_list
    else:
        remove_agents_constraints_fast(agents, vc_hard_np, ec_hard_np, pc_hard_np,0)
        for agent in agents:
            update_constraints_tracked(agent.path + agent.k_path[1:k_limit+1], vc_hard_np, ec_hard_np, pc_hard_np, agent)
        tabu_list.append(agents[0])
        return 0, tabu_list

def compute_distance(current_node, goal_node, h_dict):

    goal_heuristic_map = h_dict[goal_node.xy_name]

    dist = int(goal_heuristic_map[current_node.x, current_node.y])

    return dist

def calculate_arrival_time(agents):
    k_path_arrival_time = []
    temp_path_arrival_time = []
    diff_arr = []
    for agent in agents:
        k_counter, temp_counter = 0, 0
        for node_index in range (len(agent.k_path)):
            if agent.k_path[node_index] != agent.goal_node:
                k_counter += 1
            if node_index >= 100:
                if agent.temp_path[-1] != agent.goal_node:
                    temp_counter += 1
            else:
                if agent.temp_path[node_index] != agent.goal_node:
                    temp_counter += 1
        k_path_arrival_time.append(k_counter)
        temp_path_arrival_time.append(temp_counter)
    for i in range(len(k_path_arrival_time)):
        diff_arr.append(k_path_arrival_time[i] - temp_path_arrival_time[i])
    if any(x > 0 for x in diff_arr) and all(x >= 0 for x in diff_arr) and not all(x == 0 for x in diff_arr):
        return 0,1
    else:
        return 1,0

def calculate_arrival_time_single_agent(agent):
    k_counter = 0
    for node_index in range (len(agent.k_path)):
        if agent.k_path[node_index] != agent.goal_node:
            k_counter += 1
    return k_counter


def select_heuristics(weights, rng):
    values = {h: w['value'] for h, w in weights.items()}
    total = sum(values.values())
    prob = {h: v / total for h, v in values.items()}
    return rng.choice(list(prob.keys()), p=list(prob.values()))


def update_weights_multi_arm(heuristic, success, weights):
    learning_rate = weights[heuristic]['successes'] / weights[heuristic]['expansions']*2
    if success == True:
        weights[heuristic]['value'] = (learning_rate + 1) * weights[heuristic]['value']
    else:
        weights[heuristic]['value'] = (1 - learning_rate) * weights[heuristic]['value']

def multi_arm_resources(resources, weights,heuristic_counter, rng):
    heuristic = select_heuristics(weights,rng)
    heuristic_counter[heuristic] += 1
    if heuristic == 'Shared':
        arm_resources = resource_and_neib_distributions.create(fixed_distribution, neib_shared,
                                                           neib_agents_shared, 0, resources.num_agents)
    if heuristic == 'CPB':
        arm_resources = resource_and_neib_distributions.create(fixed_distribution, neib_proportions_with_prefix,
                                                               neib_agents_shared, 0, resources.num_agents)
    if heuristic == 'RCPB':
        arm_resources = resource_and_neib_distributions.create(fixed_distribution, neib_proportions_with_prefix_reversed,
                                                               neib_agents_shared, 0, resources.num_agents)
    if heuristic == 'PID':
        arm_resources = resource_and_neib_distributions.create(fixed_distribution, neib_pid,
                                                               neib_agents_shared, 0, resources.num_agents)
    arm_resources.max_nodes = resources.max_nodes.copy()
    arm_resources.agents_subset = resources.agent_subset
    return arm_resources, heuristic
