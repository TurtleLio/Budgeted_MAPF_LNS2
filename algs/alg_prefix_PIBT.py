from algs.real_alg_functions_pibt import *
from run_single_MAPF_func import run_mapf_alg
import matplotlib
import copy


def run_prefix_pibt(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict
) -> Tuple[None, Dict] | Tuple[Dict[str, List[Node]], Dict]:
    """
    -> LMAPF:
    - stop condition: the end of n iterations where every iteration has a time limit
    - behaviour, when agent is at its goal: agent receives a new goal
    - output: throughput
    """
    max_iter_time: int | float = params['max_iter_time']
    n_steps: int = params['n_steps']
    alg_name: bool = params['alg_name']
    to_render: bool = params['final_render']
    img_np: np.ndarray = params['img_np']
    k_limit: bool = params['k_limit']
    budget_used=[0]

    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    start_time = time.time()
    throughput: int = 0

    agents, agents_dict = create_agents(start_nodes, goal_nodes)
    n_agents = len(agents_dict)
    agents.sort(key=lambda a: a.priority, reverse=True)

    for step_iter in range(n_steps):
        if step_iter == k_limit:
            break

        config_from: Dict[str, Node] = {a.name: a.path[-1] for a in agents}
        occupied_from: Dict[str, AgentAlg] = {a.path[-1].xy_name: a for a in agents}
        config_to: Dict[str, Node] = {}
        occupied_to: Dict[str, AgentAlg] = {}

        for agent in agents:
            if agent.name not in config_to:
                _ = run_procedure_pibt(
                    agent,
                    config_from, occupied_from,
                    config_to, occupied_to,
                    agents_dict, nodes_dict, h_dict, [],budget_used)

        agents_finished, agents_unfinished = [], []
        for agent in agents:
            next_node = config_to[agent.name]
            agent.path.append(next_node)
            agent.prev_node = agent.curr_node
            agent.curr_node = next_node
            if agent.goal_node is None:
                agent.priority = agent.init_priority
                agents_finished.append(agent)
            elif agent.curr_node != agent.goal_node:
                agent.priority += 1
                agents_unfinished.append(agent)
            else:
                agent.priority = agent.init_priority
                agents_finished.append(agent)


        agents.sort(key=lambda a: a.priority, reverse=True)


        runtime = time.time() - start_time

    agents.sort(key=lambda a: a.num)
    return {a.name: a.path for a in agents}, {'agents': agents, 'throughput': throughput, 'iteration': step_iter+1}, agents

