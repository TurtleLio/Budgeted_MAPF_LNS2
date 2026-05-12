from babel.numbers import is_currency

from globals import *
from functions_general import *
from functions_plotting import *


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# CLASSES
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# FUNCS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# def get_sorted_nei_nodes(
#         agent: AgentAlg,
#         config_from: Dict[str, Node],
#         # nodes_dict: Dict[str, Node],
#         h_dict: Dict[str, np.ndarray],
# ):
#     h_goal_np: np.ndarray = h_dict[agent.get_goal_node().xy_name]
#     # sort C in ascending order of dist(u, gi) where u ∈ C
#     # nei_nodes: List[Node] = [nodes_dict[n_name] for n_name in config_from[agent.name].neighbours]
#     nei_nodes: List[Node] = config_from[agent.name].neighbours_nodes[:]
#     random.shuffle(nei_nodes)
#
#     def get_nei_v(n: Node) -> float:
#         return float(h_goal_np[n.x, n.y])
#
#     nei_nodes.sort(key=get_nei_v)
#     return nei_nodes


def get_sorted_nei_nodes(
        agent: AgentAlg,
        curr_node: Node,
        h_dict: Dict[str, np.ndarray],
):
    h_goal_np: np.ndarray = h_dict[agent.get_goal_node().xy_name]
    # sort C in ascending order of dist(u, gi) where u ∈ C
    # nei_nodes: List[Node] = [nodes_dict[n_name] for n_name in curr_node.neighbours]
    nei_nodes: List[Node] = curr_node.neighbours_nodes[:]
    random.shuffle(nei_nodes)

    def get_nei_v(n: Node) -> float:
        return float(h_goal_np[n.x, n.y])

    nei_nodes.sort(key=get_nei_v)
    return nei_nodes


def there_is_vc(
        nei_node: Node,
        config_to: Dict[str, Node],
) -> bool:
    for name, n in config_to.items():
        if nei_node == n:
            return True
    return False


def get_agent_k(
        nei_node: Node,
        occupied_from: Dict[str, AgentAlg],
        config_to: Dict[str, Node],
) -> AgentAlg | None:
    if nei_node.xy_name in occupied_from:
        other_agent = occupied_from[nei_node.xy_name]
        if other_agent.name not in config_to:
            return other_agent
    return None
    # for a_f_name, n_f_node in config_from.items():
    #     if n_f_node == nei_node and a_f_name not in config_to:
    #         return agents_dict[a_f_name]
    # return None


def there_is_ec(
        agent_i: AgentAlg,
        node_to: Node,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
) -> bool:
    node_from = config_from[agent_i.name]
    for other_name, other_node_from in config_from.items():
        if other_name == agent_i.name or other_name not in config_to:
            continue
        other_node_to = config_to[other_name]
        if other_node_from == node_to and other_node_to == node_from:
            return True
    return False


def get_next_node(node: Node, blocked: List[Node]) -> Node | None:
    nei_nodes = node.neighbours_nodes[:]
    nei_nodes.remove(node)
    for n in blocked:
        nei_nodes.remove(n)
    if len(nei_nodes) == 0:
        return None
    return random.choice(nei_nodes)

# swap_is_required = check_if_swap_required(agent_k, agent_i, i_curr_node, first_node, h_dict)
def check_if_swap_required(
        agent_i: AgentAlg,
        agent_j: AgentAlg,
        i_curr_node: Node,
        j_curr_node: Node,
        # config_from: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
) -> bool:
    """
    This is done by continuously moving i to j’s location while moving j to another vertex not equal to i’s location,
    ignoring the other agents.
    The emulation stops in two cases:
    (i) The swap is not required when j’s location has a degree of more than two.
    (ii) The swap is required when
        (1) j’s location has a degree of one,
        or,
        (2) when i reaches gi while j’s nearest neighboring vertex toward its goal is gi.
    """
    # i_curr_node = config_from[agent_i.name]
    # j_curr_node = config_from[agent_j.name]
    i_goal_node = agent_i.get_goal_node()
    if len(j_curr_node.neighbours) > 3:
        return False

    while True:

        next_node_i = j_curr_node
        next_node_j = get_next_node(j_curr_node, blocked=[i_curr_node])

        if next_node_j is None:
            return True

        if len(next_node_j.neighbours) > 3:
            return False

        if next_node_i == i_goal_node:
            nei_nodes_j = get_sorted_nei_nodes(agent_j, next_node_j, h_dict)
            nearest_nei_to_goal_j = nei_nodes_j[0]
            return nearest_nei_to_goal_j == i_goal_node

        i_curr_node = next_node_i
        j_curr_node = next_node_j


def check_if_swap_possible(
        # agent_i: AgentAlg,
        # agent_j: AgentAlg,
        i_curr_node: Node,
        j_curr_node: Node,
        # config_from: Dict[str, Node],
) -> bool:
    """
    This is done by reversing the emulation direction; that is,
    continuously moving j to i’s location while moving i to another vertex.
    It stops in two cases:
        (i) The swap is possible when i’s location has a degree of more than two.
        (ii) The swap is impossible when i is on a vertex with degree of one.
    :return:
    """
    # i_curr_node = config_from[agent_i.name]
    # j_curr_node = config_from[agent_j.name]
    while True:

        next_node_j = i_curr_node
        next_node_i = get_next_node(i_curr_node, blocked=[j_curr_node])

        if next_node_i is None:
            return False

        if len(next_node_i.neighbours) > 3:
            return True

        i_curr_node = next_node_i
        j_curr_node = next_node_j


def swap_required_and_possible(
        agent_i: AgentAlg,
        first_node: Node,
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        occupied_from: Dict[str, AgentAlg],
        h_dict: Dict[str, np.ndarray],
        with_swap: bool,
        iteration: int = 0,
) -> AgentAlg | None:
    # first_node_name = first_node.xy_name
    if not with_swap:
        return None
    i_curr_node = config_from[agent_i.name]
    if i_curr_node == first_node:
        return None
    # for the a and b cases
    if first_node.xy_name in occupied_from:
        agent_j: AgentAlg = occupied_from[first_node.xy_name]
        assert agent_j != agent_i
        if agent_j.name in config_to:
            return None
        # necessity of the swap
        j_curr_node = config_from[agent_j.name]
        # is_required = check_if_swap_required(agent_i, agent_j, config_from, h_dict)
        swap_is_required = check_if_swap_required(agent_i, agent_j, i_curr_node, j_curr_node, h_dict)
        # possibility of the swap
        i_curr_node = config_from[agent_i.name]
        j_curr_node = config_from[agent_j.name]
        swap_is_possible = check_if_swap_possible(i_curr_node, j_curr_node)
        # is_possible = check_if_swap_possible(j_curr_node, i_curr_node)
        if swap_is_required and swap_is_possible:
            return agent_j

    # for the c case
    i_curr_node = config_from[agent_i.name]
    i_nei_nodes = i_curr_node.neighbours_nodes
    for i_nei_node in i_nei_nodes:
        if i_nei_node == i_curr_node or i_nei_node.xy_name not in occupied_from or first_node == i_nei_node:
            continue
        agent_k: AgentAlg = occupied_from[i_nei_node.xy_name]
        swap_is_required = check_if_swap_required(agent_k, agent_i, i_curr_node, first_node, h_dict)
        swap_is_possible = check_if_swap_possible(i_curr_node, first_node)
        # swap_is_possible = check_if_swap_possible(first_node, i_curr_node)
        if swap_is_required and swap_is_possible:
            return agent_k
    return None


def run_procedure_pibt(
        agent_i: AgentAlg,
        config_from: Dict[str, Node],
        occupied_from: Dict[str, AgentAlg],
        config_to: Dict[str, Node],
        occupied_to: Dict[str, AgentAlg],
        agents_dict: Dict[str, AgentAlg],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        blocked_nodes_names: List[str],
        budget_used: List[int],
        iteration: int = 0,
        with_message: str = '',
        with_swap: bool = True,
        # with_swap: bool = False
) -> bool:  # valid or invalid
    agent_i.message += f'| [{iteration}-{with_message}] pibt |'

    # nei_nodes = get_sorted_nei_nodes(main_agent, config_from, nodes_dict, h_dict)
    nei_nodes = get_sorted_nei_nodes(agent_i, config_from[agent_i.name], h_dict)

    #  j ← swap_required_and_possible
    agent_j = swap_required_and_possible(agent_i, nei_nodes[0], config_from, config_to, occupied_from, h_dict, with_swap, iteration=iteration)
    if agent_j is not None:
        nei_nodes.reverse()
    for iter_nei_n, nei_node in enumerate(nei_nodes):

        if nei_node.xy_name in occupied_to:
            continue

        node_from = config_from[agent_i.name]
        if node_from.xy_name in occupied_to:
            other_agent = occupied_to[node_from.xy_name]
            if other_agent != agent_i and config_from[other_agent.name] == nei_node:
                continue

        if nei_node.xy_name in blocked_nodes_names:
            continue

        config_to[agent_i.name] = nei_node
        occupied_to[nei_node.xy_name] = agent_i

        agent_k = get_agent_k(nei_node, occupied_from, config_to)
        if agent_k is not None and agent_k != agent_i:
            valid = run_procedure_pibt(
                agent_k,
                config_from, occupied_from,
                config_to, occupied_to,
                agents_dict, nodes_dict, h_dict, blocked_nodes_names,budget_used, iteration, with_message, with_swap
            )
            if not valid:
                continue
        if with_swap and iter_nei_n == 0 and agent_j is not None and agent_j.name not in config_to:
            budget_used[-1] += 1
            i_node_from = config_from[agent_i.name]
            if i_node_from.xy_name not in occupied_to:
                config_to[agent_j.name] = i_node_from
                occupied_to[i_node_from.xy_name] = agent_j
        return True
    node_from = config_from[agent_i.name]
    config_to[agent_i.name] = node_from
    occupied_to[node_from.xy_name] = agent_i
    return False












