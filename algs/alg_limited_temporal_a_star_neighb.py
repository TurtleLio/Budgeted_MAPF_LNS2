from functions_general import *
from algs.alg_temporal_a_star_functions import *

def run_limited_temporal_a_star(
        start_node: Node,
        goal_node: Node,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        vc_hard_np: np.ndarray | None,
        ec_hard_np: np.ndarray | None,
        pc_hard_np: np.ndarray | None,
        vc_soft_np: np.ndarray | None,
        ec_soft_np: np.ndarray | None,
        pc_soft_np: np.ndarray | None,
        resources,
        max_depth: int = 100,
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent=None,
        agent_index: int = 0,
        is_neighborhood: bool = False,
        **kwargs,
) -> Tuple[List[Node] | None, dict]:
    """
    Limited A* search with temporal constraints.
    Returns the best path found that satisfies constraints, or a "wait" path if no valid path is found.
    If a "wait" path causes a collision, all previous agents' paths are reverted to "wait" paths.
    """
    start_time = time.time()
    goal_h_dict: np.ndarray = h_dict[goal_node.xy_name]
    initial_h = int(goal_h_dict[start_node.x, start_node.y])
    start_astr_node = AStarNode(start_node, 0, initial_h)
    open_list: List[AStarNode] = [start_astr_node]
    open_list_names: List[str] = [start_astr_node.xyt_name]
    closed_list_names: List[str] = []

    max_pc_time = np.max(pc_hard_np) if pc_hard_np is not None else 0
    best_node = None
    best_heuristic = float('inf')
    best_path = None

    if max_pc_time > 0:
        max_final_time = max_pc_time


    while len(open_list) > 0 and (resources.resource_subtraction(agent.num) if is_neighborhood == False else resources.neib_resource_subtraction(agent.num)):
        next_astr_node: AStarNode = heapq.heappop(open_list)
        open_list_names.remove(next_astr_node.xyt_name)

        if next_astr_node.h < best_heuristic:
            latest_vc_on_node: int = get_latest_vc_on_node(next_astr_node, vc_hard_np)
            if next_astr_node.t > latest_vc_on_node or next_astr_node.t >= max_depth:
                best_node = next_astr_node
                best_heuristic = next_astr_node.h
                best_path = reconstruct_path(best_node)

        if next_astr_node.n == goal_node and next_astr_node.t <= max_depth:
            latest_vc_on_node: int = get_latest_vc_on_node(next_astr_node, vc_hard_np)
            if next_astr_node.t > latest_vc_on_node or next_astr_node.t >= max_depth:
                path = reconstruct_path(next_astr_node)
                runtime = time.time() - start_time
                return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}

        for nei_node in next_astr_node.neighbours_nodes:
            new_t = next_astr_node.t + 1
            if new_t > max_depth:
                continue

            nei_astr_name = f'{nei_node.x}_{nei_node.y}_{new_t}'

            if nei_astr_name in open_list_names or nei_astr_name in closed_list_names:
                continue

            if new_t < vc_hard_np.shape[-1] and vc_hard_np[nei_node.x, nei_node.y, new_t] != 0:
                continue
            if new_t < ec_hard_np.shape[-1] and ec_hard_np.shape[-1] > 0:
                if ec_hard_np[next_astr_node.x, next_astr_node.y, nei_node.x, nei_node.y, new_t] != 0:
                    continue
                if ec_hard_np[nei_node.x, nei_node.y, next_astr_node.x, next_astr_node.y, new_t] != 0:
                    continue
            pc_value = pc_hard_np[nei_node.x, nei_node.y] if pc_hard_np is not None else -1
            if pc_value != -1 and new_t >= pc_value:
                continue

            new_h = int(goal_h_dict[nei_node.x, nei_node.y])
            nei_astr_node = AStarNode(nei_node, new_t, new_h, parent=next_astr_node)

            heapq.heappush(open_list, nei_astr_node)
            open_list_names.append(nei_astr_node.xyt_name)

        closed_list_names.append(next_astr_node.xyt_name)

    runtime = time.time() - start_time
    if "pie" in resources.class_name:
        return None, {'runtime': runtime}
    else:
        if best_path is not None:
            return best_path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}
        else:
            wait_path = [start_node] * (max_depth + 1)
            return wait_path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}


def causes_collision(path, vc_hard_np, ec_hard_np, pc_hard_np):
    """
    Checks if a path causes any collision based on the current constraints.
    """
    for t, n in enumerate(path):
        if t < vc_hard_np.shape[-1] and vc_hard_np[n.x, n.y, t]:
            return True
    return False


def collides_with_existing_paths(path: List[Node], existing_paths: List[List[Tuple[int, int]]]) -> bool:
    """
    Checks if the given path collides with any existing paths.
    """
    for t, node in enumerate(path):
        pos = (node.x, node.y)
        for other_path in existing_paths:
            if t < len(other_path) and pos == other_path[t]:
                return True
    return False

def check_for_collisions(all_agents_paths):
    """
    Checks for collisions between the paths of all agents.
    Prints out details of any detected collisions.

    :param all_agents_paths: List of paths for all agents.
                             Each path is a list of nodes (x, y) at each time step.
    """
    collision_found = False
    occupancy = {}

    for agent_index, path in enumerate(all_agents_paths):
        for t, node in enumerate(path):
            position_time = (node.x, node.y, t)
            if position_time in occupancy:
                collision_found = True
                occupancy[position_time].append(agent_index)
            else:
                occupancy[position_time] = [agent_index]


def run_limited_temporal_a_star_optimized_LNS1(
        start_node: Node,
        goal_node: Node,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        vc_hard_np: np.ndarray | None,
        ec_hard_np: np.ndarray | None,
        pc_hard_np: np.ndarray | None,
        vc_soft_np: np.ndarray | None,
        ec_soft_np: np.ndarray | None,
        pc_soft_np: np.ndarray | None,
        resources,
        max_depth: int = 500,
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent=None,
        agent_index: int = 0,
        is_neighborhood: bool = False,
        path_length: int = 101,
        step_iter: int=0,
        **kwargs,
) -> Tuple[List[Node] | None, dict]:
    """
    Optimized A* that uses new fast methods but falls back to old ones
    """
    if step_iter > 1:
        step_iter -= 1
    start_time = time.time()
    try:
        initial_h = start_node.get_cached_heuristic(goal_node.xy_name, h_dict)
    except:
        goal_h_dict: np.ndarray = h_dict[goal_node.xy_name]
        initial_h = int(goal_h_dict[start_node.x, start_node.y])

    start_astr_node = AStarNode(start_node, 0, initial_h)
    open_list: List[AStarNode] = [start_astr_node]

    open_set_ids: Set[int] = {start_astr_node.state_id}
    open_list_names: List[str] = [start_astr_node.xyt_name]
    closed_set_ids: Set[int] = set()
    closed_list_names: List[str] = []

    max_vc_time = vc_hard_np.shape[-1] if vc_hard_np is not None else 0
    max_ec_time = ec_hard_np.shape[-1] if ec_hard_np is not None else 0
    max_constraint_time = min(max_vc_time, max_ec_time) if max_ec_time > 0 else max_vc_time

    goal_reached_at = None
    best_goal_node = None

    while len(open_list) > 0 and (
            resources.resource_subtraction(
                agent.num) if is_neighborhood == False else resources.neib_resource_subtraction(
                agent.num)):

        next_astr_node: AStarNode = heapq.heappop(open_list)

        try:
            open_set_ids.remove(next_astr_node.state_id)
        except:
            pass
        if next_astr_node.xyt_name in open_list_names:
            open_list_names.remove(next_astr_node.xyt_name)

        if next_astr_node.n.xy_name == goal_node.xy_name:

            future_constraints = check_future_goal_constraints(
                goal_node, next_astr_node.t, path_length, step_iter, vc_hard_np, pc_hard_np, agent
            )
            if not future_constraints:
                path = construct_goal_filled_path(next_astr_node, goal_node, path_length)
                runtime = time.time() - start_time
                return path, {'runtime': runtime, 'future_constraints': False}

        for nei_node in next_astr_node.neighbours_nodes:
            new_t = next_astr_node.t + 1
            global_new_t = new_t + step_iter


            if global_new_t > max_depth:
                continue


            nei_state_id = (nei_node.id << 16) | global_new_t

            if nei_state_id in open_set_ids or nei_state_id in closed_set_ids:
                continue



            if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                continue
            try:
                new_h = nei_node.get_cached_heuristic(goal_node.xy_name, h_dict)
            except:
                goal_h_dict = h_dict[goal_node.xy_name]
                new_h = int(goal_h_dict[nei_node.x, nei_node.y])

            nei_astr_node = AStarNode(nei_node, new_t, new_h, parent=next_astr_node)
            heapq.heappush(open_list, nei_astr_node)
            open_set_ids.add(nei_state_id)
            open_list_names.append(nei_astr_node.xyt_name)

        closed_set_ids.add(next_astr_node.state_id)
        closed_list_names.append(next_astr_node.xyt_name)


    runtime = time.time() - start_time
    return None, {'runtime': runtime}


def run_limited_temporal_a_star_optimized(
        start_node: Node,
        goal_node: Node,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        vc_hard_np: np.ndarray | None,
        ec_hard_np: np.ndarray | None,
        pc_hard_np: np.ndarray | None,
        vc_soft_np: np.ndarray | None,
        ec_soft_np: np.ndarray | None,
        pc_soft_np: np.ndarray | None,
        resources,
        max_depth: int = 500,
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent=None,
        agent_index: int = 0,
        is_neighborhood: bool = False,
        path_length: int = 100,
        step_iter: int=0,
        **kwargs,
) -> Tuple[List[Node] | None, dict]:
    """
    Optimized A* that uses new fast methods but falls back to old ones
    """
    start_time = time.time()
    try:
        initial_h = start_node.get_cached_heuristic(goal_node.xy_name, h_dict)
    except:
        goal_h_dict: np.ndarray = h_dict[goal_node.xy_name]
        initial_h = int(goal_h_dict[start_node.x, start_node.y])

    start_astr_node = AStarNode(start_node, 0, initial_h)
    open_list: List[AStarNode] = [start_astr_node]

    open_set_ids: Set[int] = {start_astr_node.state_id}
    open_list_names: List[str] = [start_astr_node.xyt_name]
    closed_set_ids: Set[int] = set()
    closed_list_names: List[str] = []

    max_vc_time = vc_hard_np.shape[-1] if vc_hard_np is not None else 0
    max_ec_time = ec_hard_np.shape[-1] if ec_hard_np is not None else 0
    max_constraint_time = min(max_vc_time, max_ec_time) if max_ec_time > 0 else max_vc_time

    goal_reached_at = None
    best_goal_node = None

    while len(open_list) > 0 and (
            resources.resource_subtraction(
                agent.num) if is_neighborhood == False else resources.neib_resource_subtraction(
                agent.num)):

        next_astr_node: AStarNode = heapq.heappop(open_list)


        try:
            open_set_ids.remove(next_astr_node.state_id)
        except:
            pass
        if next_astr_node.xyt_name in open_list_names:
            open_list_names.remove(next_astr_node.xyt_name)

        if next_astr_node.n == goal_node:

            future_constraints = check_future_goal_constraints(
                goal_node, next_astr_node.t, path_length, step_iter, vc_hard_np, pc_hard_np, agent
            )

            if not future_constraints:
                path = construct_goal_filled_path(next_astr_node, goal_node, path_length)
                runtime = time.time() - start_time
                return path, {'runtime': runtime, 'future_constraints': False}

        for nei_node in next_astr_node.neighbours_nodes:
            new_t = next_astr_node.t + 1
            global_new_t = new_t + step_iter

            if global_new_t > path_length or global_new_t >= max_constraint_time:
                continue


            nei_state_id = (nei_node.id << 16) | global_new_t

            if nei_state_id in open_set_ids or nei_state_id in closed_set_ids:
                continue



            if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                continue
            try:
                new_h = nei_node.get_cached_heuristic(goal_node.xy_name, h_dict)
            except:
                goal_h_dict = h_dict[goal_node.xy_name]
                new_h = int(goal_h_dict[nei_node.x, nei_node.y])

            nei_astr_node = AStarNode(nei_node, new_t, new_h, parent=next_astr_node)
            heapq.heappush(open_list, nei_astr_node)
            open_set_ids.add(nei_state_id)
            open_list_names.append(nei_astr_node.xyt_name)

        closed_set_ids.add(next_astr_node.state_id)
        closed_list_names.append(next_astr_node.xyt_name)


    runtime = time.time() - start_time
    return None, {'runtime': runtime}

def is_constrained_fast(current_node: AStarNode, next_node: Node, global_new_t: int,
                        vc_hard_np, ec_hard_np, pc_hard_np) -> bool:
    """Fixed constraint checking with proper bounds checking"""

    if (vc_hard_np is not None and global_new_t < vc_hard_np.shape[2]):
        constraint_value = vc_hard_np[next_node.x, next_node.y, global_new_t]
        if constraint_value != -1:
            return True

    if (ec_hard_np is not None and
            global_new_t < ec_hard_np.shape[4] and
            current_node.n.x < ec_hard_np.shape[0] and
            current_node.n.y < ec_hard_np.shape[1] and
            next_node.x < ec_hard_np.shape[2] and
            next_node.y < ec_hard_np.shape[3]):

        if ec_hard_np[current_node.n.x, current_node.n.y, next_node.x, next_node.y, global_new_t] != -1:
            return True
        if ec_hard_np[next_node.x, next_node.y, current_node.n.x, current_node.n.y, global_new_t] != -1:
            return True

    if (pc_hard_np is not None and
            next_node.x < pc_hard_np.shape[0] and
            next_node.y < pc_hard_np.shape[1]):

        pc_value = pc_hard_np[next_node.x, next_node.y]
        if pc_value != -1 and global_new_t >= pc_value:
            return True

    return False


def reconstruct_path_to_goal_then_stay(goal_node: AStarNode, goal_position: Node, target_length: int,
                                       vc_hard_np, ec_hard_np, pc_hard_np, agent) -> List[Node]:
    """
    Reconstruct path to goal, then extend by staying at goal (if not constrained)
    """
    path: List[Node] = []
    node_current = goal_node
    while node_current is not None:
        path.append(node_current.n)
        node_current = node_current.parent
    path.reverse()

    goal_reached_at = len(path) - 1

    while len(path) < target_length:
        next_time = len(path)

        if (vc_hard_np is not None and
                next_time < vc_hard_np.shape[2] and
                vc_hard_np[goal_position.x, goal_position.y, next_time] != 0 and
                vc_hard_np[goal_position.x, goal_position.y, next_time] != (agent.num if agent else -1)):
            break

        path.append(goal_position)

    return path


def check_future_goal_constraints(goal_node, current_time, path_length, step_iter, vc_hard_np, pc_hard_np, agent):
    """Check if there are any future constraints at the goal position"""
    future_constraints = []

    for local_t in range(current_time + 1, path_length):
        global_t = local_t + step_iter

        if (vc_hard_np is not None and
                global_t < vc_hard_np.shape[2] and
                vc_hard_np[goal_node.x, goal_node.y, global_t] != -1):
            future_constraints.append(local_t)

    return future_constraints


def construct_goal_filled_path(goal_node_astar, goal_node, path_length):
    """Construct path to goal, then fill remaining with goal node"""
    path = []
    node_current = goal_node_astar
    while node_current is not None:
        path.append(node_current.n)
        node_current = node_current.parent
    path.reverse()

    while len(path) < path_length:
        path.append(goal_node)

    return path[:path_length]


def run_limited_recursive_a_star(
        start_node: Node,
        goal_node: Node,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        vc_hard_np: np.ndarray | None,
        ec_hard_np: np.ndarray | None,
        pc_hard_np: np.ndarray | None,
        vc_soft_np: np.ndarray | None,
        ec_soft_np: np.ndarray | None,
        pc_soft_np: np.ndarray | None,
        resources,
        max_depth: int = 500,
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent=None,
        agent_index: int = 0,
        is_neighborhood: bool = False,
        path_length: int = 100,
        step_iter: int=0,
        **kwargs,
) -> Tuple[List[Node] | None, dict]:
    """
    Optimized A* that uses new fast methods but falls back to old ones
    """
    start_time = time.time()
    try:
        initial_h = start_node.get_cached_heuristic(goal_node.xy_name, h_dict)
    except:
        goal_h_dict: np.ndarray = h_dict[goal_node.xy_name]
        initial_h = int(goal_h_dict[start_node.x, start_node.y])

    start_astr_node = AStarNode(start_node, 0, initial_h)
    open_list: List[AStarNode] = [start_astr_node]

    open_set_ids: Set[int] = {start_astr_node.state_id}
    open_list_names: List[str] = [start_astr_node.xyt_name]
    closed_set_ids: Set[int] = set()
    closed_list_names: List[str] = []

    max_vc_time = vc_hard_np.shape[-1] if vc_hard_np is not None else 0
    max_ec_time = ec_hard_np.shape[-1] if ec_hard_np is not None else 0
    max_constraint_time = min(max_vc_time, max_ec_time) if max_ec_time > 0 else max_vc_time

    goal_reached_at = None
    best_goal_node = None

    while len(open_list) > 0 and (
            resources.resource_subtraction(
                agent.num) if is_neighborhood == False else resources.neib_resource_subtraction(
                agent.num)):

        next_astr_node: AStarNode = heapq.heappop(open_list)

        try:
            open_set_ids.remove(next_astr_node.state_id)
        except:
            pass
        if next_astr_node.xyt_name in open_list_names:
            open_list_names.remove(next_astr_node.xyt_name)


        if next_astr_node.n == goal_node:
            path = recursive_construct_path(
                goal_node, next_astr_node.t, path_length, step_iter, vc_hard_np, pc_hard_np, agent
            )
            runtime = time.time() - start_time
            return path, {'runtime': runtime, 'future_constraints': False}


        for nei_node in next_astr_node.neighbours_nodes:
            new_t = next_astr_node.t + 1
            global_new_t = new_t + step_iter


            if global_new_t > path_length or global_new_t >= max_constraint_time:
                continue

            nei_state_id = (nei_node.id << 16) | global_new_t

            if nei_state_id in open_set_ids or nei_state_id in closed_set_ids:
                continue

            if goal_reached_at is not None and global_new_t > goal_reached_at:
                if nei_node != goal_node:
                    continue

                if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                    continue

            if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                continue

            try:
                new_h = nei_node.get_cached_heuristic(goal_node.xy_name, h_dict)
            except:
                goal_h_dict = h_dict[goal_node.xy_name]
                new_h = int(goal_h_dict[nei_node.x, nei_node.y])

            nei_astr_node = AStarNode(nei_node, new_t, new_h, parent=next_astr_node)

            heapq.heappush(open_list, nei_astr_node)
            open_set_ids.add(nei_state_id)
            open_list_names.append(nei_astr_node.xyt_name)

        closed_set_ids.add(next_astr_node.state_id)
        closed_list_names.append(next_astr_node.xyt_name)

    runtime = time.time() - start_time
    return None, {'runtime': runtime}


def recursive_construct_path(goal_node: AStarNode, goal_position: Node, target_length: int,
                                       vc_hard_np, ec_hard_np, pc_hard_np, agent,         start_node: Node,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        resources,
        max_depth: int = 500,
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent_index: int = 0,
        is_neighborhood: bool = False,
        path_length: int = 100,
        step_iter: int=0,):
    path: List[Node] = []
    node_current = goal_node
    while node_current is not None:
        path.append(node_current.n)
        node_current = node_current.parent
    path.reverse()

    goal_reached_at = len(path) - 1

    while len(path) < target_length:
        next_time = len(path)
        next_node = goal_position
        if (vc_hard_np is not None and
                next_time < vc_hard_np.shape[2] and
                vc_hard_np[goal_position.x, goal_position.y, next_time] != 0 and
                vc_hard_np[goal_position.x, goal_position.y, next_time] != (agent.num if agent else -1)):
            next_node = run_limited_recursive_a_star(path[-1], agent.goal_node, nodes, nodes_dict, h_dict,
            vc_hard_np, ec_hard_np, pc_hard_np, resources, max_depth=max_depth-len(path),
            agent=agent, is_neighborhood = True, step_iter=len(path))
            if next_node == None:
                return None

        path.append(next_node)
    if path[-1] == goal_node:
        return path
    else:
        return None
