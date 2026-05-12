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

    # create agents
    agents, agents_dict = create_agents(start_nodes, goal_nodes)
    n_agents = len(agents_dict)
    #agents_copy = copy.deepcopy(agents)
    agents.sort(key=lambda a: a.priority, reverse=True)

    for step_iter in range(n_steps):
        if step_iter == k_limit:
            break

        config_from: Dict[str, Node] = {a.name: a.path[-1] for a in agents}
        occupied_from: Dict[str, AgentAlg] = {a.path[-1].xy_name: a for a in agents}
        config_to: Dict[str, Node] = {}
        occupied_to: Dict[str, AgentAlg] = {}

        # calc the step
        for agent in agents:
            if agent.name not in config_to:
                _ = run_procedure_pibt(
                    agent,
                    config_from, occupied_from,
                    config_to, occupied_to,
                    agents_dict, nodes_dict, h_dict, [],budget_used)

        # execute the step + check the termination condition
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


        # unfinished first
        agents.sort(key=lambda a: a.priority, reverse=True)
        # agents_unfinished.sort(key=lambda a: a.priority, reverse=True)
        # agents = [*agents_finished, *agents_unfinished]

        # throughput += update_goal_nodes(agents, nodes)

        # print + render
        runtime = time.time() - start_time
        #print(f'\r[{alg_name}] {step_iter=: <3} | runtime: {runtime: .2f} s. | throughput: {throughput=} | iteration{step_iter}' , end='')
        # ------------------------------ #
        # ------------------------------ #
        # ------------------------------ #
        # budget_used[-1] = budget_used[-1]
        # budget_used.append(0)
        # print(f"budget_used in PIBT: {budget_used}")
        # if to_render:
        #     # plot the iteration
        #     i_agent = agents[0]
        #     plot_info = {
        #         'img_np': img_np,
        #         'agents': agents,
        #         'i_agent': i_agent,
        #         'i': step_iter,
        #     }
        #     plot_step_in_env(ax[0], plot_info)
        #     plt.pause(0.001)
        #     # plt.pause(1)
        # if 'Lifelong' in alg_name:
        #     throughput += update_goal_nodes(agents, nodes)
        # if 'k' in alg_name and all_agents_reached_goal(agents):
        #     break

    # checks
    # for i in range(len(agents[0].path)):
    #     check_vc_ec_neic_iter(agents, i, to_count=False)
    #print(f"throughput: {throughput:.2f}")
    #     budget_per_step = []
    # budget_per_5_steps = []
    # for budget in budget_used:
    #     budget_per_step.append(budget / len(agents))
    #     if len(budget_used) % 5 != 0:
    #         budget_per_5_steps.append(0)
    # for i in range(int(len(budget_used) / 5)):
    #     budget_per_5_steps.append(((budget_used[5 * i] + budget_used[5 * i + 1] + budget_used[5 * i + 2] +
    #                                 budget_used[5 * i + 3] + budget_used[5 * i + 4]) / 5) / len(agents))
    # print(f"finished in {step_iter} steps")
    # print(f'budget per step: {budget_per_step}')
    # print(f'budget per 5 steps: {budget_per_5_steps}')
    agents.sort(key=lambda a: a.num)
    return {a.name: a.path for a in agents}, {'agents': agents, 'throughput': throughput, 'iteration': step_iter+1}, agents

