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
        vc_soft_np: np.ndarray | None,  # Optional soft constraints (currently unused)
        ec_soft_np: np.ndarray | None,
        pc_soft_np: np.ndarray | None,
        resources,  # The class that contains the limit of the nodes
        max_depth: int = 100,  # Limit on the depth of the search
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent=None,
        agent_index: int = 0,   # Current agent index
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
    best_heuristic = float('inf')  # Lower heuristic is better
    best_path = None

    if max_pc_time > 0:
        max_final_time = max_pc_time


    while len(open_list) > 0 and (resources.resource_subtraction(agent.num) if is_neighborhood == False else resources.neib_resource_subtraction(agent.num)):
        next_astr_node: AStarNode = heapq.heappop(open_list)
        open_list_names.remove(next_astr_node.xyt_name)

        # Update best node if this node has a better heuristic and satisfies constraints
        if next_astr_node.h < best_heuristic:
            latest_vc_on_node: int = get_latest_vc_on_node(next_astr_node, vc_hard_np)
            if next_astr_node.t > latest_vc_on_node or next_astr_node.t >= max_depth:
                best_node = next_astr_node
                best_heuristic = next_astr_node.h
                best_path = reconstruct_path(best_node)

        # Check if the goal is reached and satisfies constraints
        if next_astr_node.n == goal_node and next_astr_node.t <= max_depth:
            latest_vc_on_node: int = get_latest_vc_on_node(next_astr_node, vc_hard_np)
            if next_astr_node.t > latest_vc_on_node or next_astr_node.t >= max_depth:
                path = reconstruct_path(next_astr_node)
                runtime = time.time() - start_time
                # Update constraints with the valid path and return it
                #update_constraints(path, vc_hard_np, ec_hard_np, pc_hard_np, agent.num)
                return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}

        # Explore neighbors
        for nei_node in next_astr_node.neighbours_nodes:
            new_t = next_astr_node.t + 1
            if new_t > max_depth:
                continue

            nei_astr_name = f'{nei_node.x}_{nei_node.y}_{new_t}'

            if nei_astr_name in open_list_names or nei_astr_name in closed_list_names:
                continue

            # Check constraints
            # if new_t < vc_hard_np.shape[-1] and vc_hard_np[nei_node.x, nei_node.y, new_t]:
            #     continue
            if new_t < vc_hard_np.shape[-1] and vc_hard_np[nei_node.x, nei_node.y, new_t] != 0:
                continue
            # if new_t < ec_hard_np.shape[-1] and ec_hard_np[nei_node.x, nei_node.y, next_astr_node.x, next_astr_node.y, new_t]:
            #     continue
            if new_t < ec_hard_np.shape[-1] and ec_hard_np.shape[-1] > 0:
                # Check forward edge
                if ec_hard_np[next_astr_node.x, next_astr_node.y, nei_node.x, nei_node.y, new_t] != 0:
                    continue
                # Check reverse edge (head-on collision)
                if ec_hard_np[nei_node.x, nei_node.y, next_astr_node.x, next_astr_node.y, new_t] != 0:
                    continue
            pc_value = pc_hard_np[nei_node.x, nei_node.y] if pc_hard_np is not None else -1
            if pc_value != -1 and new_t >= pc_value:
                continue

            # Compute heuristic and create new AStarNode
            new_h = int(goal_h_dict[nei_node.x, nei_node.y])
            nei_astr_node = AStarNode(nei_node, new_t, new_h, parent=next_astr_node)

            heapq.heappush(open_list, nei_astr_node)
            open_list_names.append(nei_astr_node.xyt_name)

        # Add current node to closed list
        closed_list_names.append(next_astr_node.xyt_name)

    # If no valid path is found within the limits
    runtime = time.time() - start_time
    if "pie" in resources.class_name:
        return None, {'runtime': runtime}
    else:
        if best_path is not None:
            # Save the best path that satisfies constraints
            #update_constraints(best_path, vc_hard_np, ec_hard_np, pc_hard_np)
            return best_path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}
        else:
            # No valid path; return "wait" path (stay in place)
            wait_path = [start_node] * (max_depth + 1)
            # update_constraints(wait_path, vc_hard_np, ec_hard_np, pc_hard_np)
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
    # Dictionary to track occupancy: key = (x, y, t), value = list of agent indices
    occupancy = {}

    for agent_index, path in enumerate(all_agents_paths):
        for t, node in enumerate(path):
            position_time = (node.x, node.y, t)
            if position_time in occupancy:
                # Collision detected
                collision_found = True
                #print(f"Collision detected at (x={node.x}, y={node.y}, t={t}) between agents "
                 #     f"{occupancy[position_time]} and {agent_index}")
                occupancy[position_time].append(agent_index)
            else:
                occupancy[position_time] = [agent_index]

    # if not collision_found:
    #     print("No collisions detected.")

def run_limited_temporal_a_star_optimized_LNS1(
        start_node: Node,
        goal_node: Node,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        vc_hard_np: np.ndarray | None,
        ec_hard_np: np.ndarray | None,
        pc_hard_np: np.ndarray | None,
        vc_soft_np: np.ndarray | None,  # Optional soft constraints (currently unused)
        ec_soft_np: np.ndarray | None,
        pc_soft_np: np.ndarray | None,
        resources,  # The class that contains the limit of the nodes
        max_depth: int = 500,  # Limit on the depth of the search
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent=None,
        agent_index: int = 0,  # Current agent index
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

    # 🚀 Use sets for faster lookups, but keep string names as backup
    open_set_ids: Set[int] = {start_astr_node.state_id}  # Fast integer lookup
    open_list_names: List[str] = [start_astr_node.xyt_name]  # Backward compatibility
    closed_set_ids: Set[int] = set()  # Fast integer lookup
    closed_list_names: List[str] = []  # Backward compatibility

    # 🚀 Pre-compute constraint bounds (optimization)
    max_vc_time = vc_hard_np.shape[-1] if vc_hard_np is not None else 0
    max_ec_time = ec_hard_np.shape[-1] if ec_hard_np is not None else 0
    max_constraint_time = min(max_vc_time, max_ec_time) if max_ec_time > 0 else max_vc_time

    # Track when goal is reached
    goal_reached_at = None
    best_goal_node = None

    while len(open_list) > 0 and (
            resources.resource_subtraction(
                agent.num) if is_neighborhood == False else resources.neib_resource_subtraction(
                agent.num)):

        next_astr_node: AStarNode = heapq.heappop(open_list)
        # print(f"next_node:{next_astr_node.n} | goal_node:{goal_node}")
        # Debug code (keeping your existing debug)
        # if agent and agent.num == 68:
        #     print(f"🔍 DEBUGGING A* for Agent {agent.num}")
        #     print(f"  Start: ({start_node.x},{start_node.y})")
        #     print(f"  Goal: ({goal_node.x},{goal_node.y})")
        #     print(f"  Current node: ({agent.curr_node.x},{agent.curr_node.y})")
        #
        #     if vc_hard_np is not None and vc_hard_np.shape[2] > 3:
        #         constraint_22_8_t3 = vc_hard_np[22, 8, 3]
        #         print(f"  Constraint at (22,8) time 3: {constraint_22_8_t3}")
        #         if constraint_22_8_t3 != 0:
        #             print(f"  ❌ Position (22,8) at time 3 is BLOCKED by agent {constraint_22_8_t3}")

        # 🚀 Use fast integer lookup, fall back to string if needed
        try:
            open_set_ids.remove(next_astr_node.state_id)
        except:
            pass
        if next_astr_node.xyt_name in open_list_names:
            open_list_names.remove(next_astr_node.xyt_name)

        # # ✅ NEW: Check if we've reached the goal
        # if next_astr_node.n == goal_node and goal_reached_at is None:
        #     goal_reached_at = next_astr_node.t
        #     best_goal_node = next_astr_node
        #     #print(f"🎯 Agent {agent.num if agent else 'X'} reached goal at time {goal_reached_at}")
        # 🎯 GOAL REACHED - CHECK FUTURE CONSTRAINTS
        # if next_astr_node.n == goal_node:
        if next_astr_node.n.xy_name == goal_node.xy_name:
            #print(f"🎯 Agent {agent.num} reached goal at time {next_astr_node.t}")

            # Check if there are future constraints at goal
            future_constraints = check_future_goal_constraints(
                goal_node, next_astr_node.t, path_length, step_iter, vc_hard_np, pc_hard_np, agent
            )
            if not future_constraints:
                # ✅ NO FUTURE CONSTRAINTS - Fill remaining path with goal and return
                #print(f"✅ No future constraints - filling path with goal")
                path = construct_goal_filled_path(next_astr_node, goal_node, path_length)
                runtime = time.time() - start_time
                return path, {'runtime': runtime, 'future_constraints': False}
            # else:
            #     # ❌ FUTURE CONSTRAINTS EXIST - Continue searching
            #     print(f"⚠️ Future constraints found at times: {future_constraints}")
            #     print(f"🔄 Continuing search to handle constraints...")
                # Continue with normal A* exploration
        # ✅ NEW: If we've reached the desired path length, construct and return path
        # if next_astr_node.t >= path_length:
        #     if best_goal_node is not None:
        #         # Construct path: reach goal + stay at goal
        #         path = reconstruct_path_to_goal_then_stay(best_goal_node, goal_node, path_length,
        #                                                   vc_hard_np, ec_hard_np, pc_hard_np, agent)
        #         runtime = time.time() - start_time
        #         return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}
        #     else:
        #         # Haven't reached goal yet, continue searching
        #         pass

        # Explore neighbors
        for nei_node in next_astr_node.neighbours_nodes:
            new_t = next_astr_node.t + 1
            global_new_t = new_t + step_iter

            # Debug: Check if this neighbor would move us closer to goal
            # if agent and agent.num == 3:  # Replace with actual agent number
            #     curr_dist = abs(next_astr_node.n.x - goal_node.x) + abs(next_astr_node.n.y - goal_node.y)
            #     nei_dist = abs(nei_node.x - goal_node.x) + abs(nei_node.y - goal_node.y)
            #
            #     # If this neighbor is the goal or moves us closer
            #     if nei_dist < curr_dist or (nei_node.x == goal_node.x and nei_node.y == goal_node.y):
            #         # Check if it's being rejected
            #         if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
            #             if nei_node.x == 10 and nei_node.y == 24:
            #                 pc_value = pc_hard_np[nei_node.x, nei_node.y] if pc_hard_np is not None else -1
            #                 print(f"  Checking goal (10,24) at time {global_new_t}, pc_value={pc_value}")
            #                 if pc_value != -1 and global_new_t >= pc_value:
            #                     print(f"  ❌ BLOCKED by precedence constraint!")
            #             print(
            #                 f"  ❌ Agent {agent.num} blocked from moving to ({nei_node.x},{nei_node.y}) at time {global_new_t}")
            #             print(f"     (would have reduced distance from {curr_dist} to {nei_dist})")

            if global_new_t > max_depth:
                continue

            # Debug code (keeping your existing debug)
            # if agent and agent.num == 68 and nei_node.x == 22 and nei_node.y == 8 and global_new_t == 3:
            #     print(f"🔍 Agent 68 trying to move to (22,8) at time 3!")
            #     if vc_hard_np is not None:
            #         constraint_value = vc_hard_np[22, 8, 3]
            #         print(f"  Constraint value at (22,8) time 3: {constraint_value}")
            #         if constraint_value != 0:
            #             print(f"  ❌ SHOULD BE BLOCKED by agent {constraint_value}")
            #         else:
            #             print(f"  ✅ No constraint found - this is the BUG!")

            # 🚀 Create state ID for fast lookup
            nei_state_id = (nei_node.id << 16) | global_new_t

            # 🚀 Fast lookup using sets (O(1) instead of O(n))
            if nei_state_id in open_set_ids or nei_state_id in closed_set_ids:
                continue

            # # ✅ NEW: If we've reached goal, only allow staying at goal (unless constrained)
            # if goal_reached_at is not None and global_new_t > goal_reached_at:
            #     if nei_node != goal_node:
            #         continue  # Must stay at goal after reaching it

                # #Check if staying at goal is constrained
                # if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                #     # print(
                #     #     f"⚠️ Agent {agent.num if agent else 'X'} cannot stay at goal at time {global_new_t} due to constraints")
                #     continue  # Cannot stay at goal due to constraints

            # 🚀 Optimized constraint checking for normal movement
            if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                continue
            try:
                new_h = nei_node.get_cached_heuristic(goal_node.xy_name, h_dict)
            except:
                goal_h_dict = h_dict[goal_node.xy_name]
                new_h = int(goal_h_dict[nei_node.x, nei_node.y])

            nei_astr_node = AStarNode(nei_node, new_t, new_h, parent=next_astr_node)
            #nei_astr_node = AStarNode(nei_node, global_new_t, new_h, parent=next_astr_node)
            heapq.heappush(open_list, nei_astr_node)
            open_set_ids.add(nei_state_id)
            open_list_names.append(nei_astr_node.xyt_name)

        # Add to closed list
        closed_set_ids.add(next_astr_node.state_id)
        closed_list_names.append(next_astr_node.xyt_name)

    # If we exit the loop, try to return best path found
    # if best_goal_node is not None:
    #     path = reconstruct_path_to_goal_then_stay(best_goal_node, goal_node, path_length,
    #                                               vc_hard_np, ec_hard_np, pc_hard_np, agent)
    #     runtime = time.time() - start_time
    #     return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}

    # Return same format as original
    # if sum(resources.max_nodes) <= 0:
    #     print(f"no more resources")
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
        vc_soft_np: np.ndarray | None,  # Optional soft constraints (currently unused)
        ec_soft_np: np.ndarray | None,
        pc_soft_np: np.ndarray | None,
        resources,  # The class that contains the limit of the nodes
        max_depth: int = 500,  # Limit on the depth of the search
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent=None,
        agent_index: int = 0,  # Current agent index
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

    # 🚀 Use sets for faster lookups, but keep string names as backup
    open_set_ids: Set[int] = {start_astr_node.state_id}  # Fast integer lookup
    open_list_names: List[str] = [start_astr_node.xyt_name]  # Backward compatibility
    closed_set_ids: Set[int] = set()  # Fast integer lookup
    closed_list_names: List[str] = []  # Backward compatibility

    # 🚀 Pre-compute constraint bounds (optimization)
    max_vc_time = vc_hard_np.shape[-1] if vc_hard_np is not None else 0
    max_ec_time = ec_hard_np.shape[-1] if ec_hard_np is not None else 0
    max_constraint_time = min(max_vc_time, max_ec_time) if max_ec_time > 0 else max_vc_time

    # Track when goal is reached
    goal_reached_at = None
    best_goal_node = None

    while len(open_list) > 0 and (
            resources.resource_subtraction(
                agent.num) if is_neighborhood == False else resources.neib_resource_subtraction(
                agent.num)):

        next_astr_node: AStarNode = heapq.heappop(open_list)

        # Debug code (keeping your existing debug)
        # if agent and agent.num == 68:
        #     print(f"🔍 DEBUGGING A* for Agent {agent.num}")
        #     print(f"  Start: ({start_node.x},{start_node.y})")
        #     print(f"  Goal: ({goal_node.x},{goal_node.y})")
        #     print(f"  Current node: ({agent.curr_node.x},{agent.curr_node.y})")
        #
        #     if vc_hard_np is not None and vc_hard_np.shape[2] > 3:
        #         constraint_22_8_t3 = vc_hard_np[22, 8, 3]
        #         print(f"  Constraint at (22,8) time 3: {constraint_22_8_t3}")
        #         if constraint_22_8_t3 != 0:
        #             print(f"  ❌ Position (22,8) at time 3 is BLOCKED by agent {constraint_22_8_t3}")

        # 🚀 Use fast integer lookup, fall back to string if needed
        try:
            open_set_ids.remove(next_astr_node.state_id)
        except:
            pass
        if next_astr_node.xyt_name in open_list_names:
            open_list_names.remove(next_astr_node.xyt_name)

        # # ✅ NEW: Check if we've reached the goal
        # if next_astr_node.n == goal_node and goal_reached_at is None:
        #     goal_reached_at = next_astr_node.t
        #     best_goal_node = next_astr_node
        #     #print(f"🎯 Agent {agent.num if agent else 'X'} reached goal at time {goal_reached_at}")
        # 🎯 GOAL REACHED - CHECK FUTURE CONSTRAINTS
        if next_astr_node.n == goal_node:
            #print(f"🎯 Agent {agent.num} reached goal at time {next_astr_node.t}")

            # Check if there are future constraints at goal
            future_constraints = check_future_goal_constraints(
                goal_node, next_astr_node.t, path_length, step_iter, vc_hard_np, pc_hard_np, agent
            )

            if not future_constraints:
                # ✅ NO FUTURE CONSTRAINTS - Fill remaining path with goal and return
                #print(f"✅ No future constraints - filling path with goal")
                path = construct_goal_filled_path(next_astr_node, goal_node, path_length)
                runtime = time.time() - start_time
                return path, {'runtime': runtime, 'future_constraints': False}
            # else:
            #     # ❌ FUTURE CONSTRAINTS EXIST - Continue searching
            #     print(f"⚠️ Future constraints found at times: {future_constraints}")
            #     print(f"🔄 Continuing search to handle constraints...")
                # Continue with normal A* exploration
        # ✅ NEW: If we've reached the desired path length, construct and return path
        # if next_astr_node.t >= path_length:
        #     if best_goal_node is not None:
        #         # Construct path: reach goal + stay at goal
        #         path = reconstruct_path_to_goal_then_stay(best_goal_node, goal_node, path_length,
        #                                                   vc_hard_np, ec_hard_np, pc_hard_np, agent)
        #         runtime = time.time() - start_time
        #         return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}
        #     else:
        #         # Haven't reached goal yet, continue searching
        #         pass

        # Explore neighbors
        for nei_node in next_astr_node.neighbours_nodes:
            new_t = next_astr_node.t + 1
            global_new_t = new_t + step_iter

            if global_new_t > path_length or global_new_t >= max_constraint_time:
                continue

            # Debug code (keeping your existing debug)
            # if agent and agent.num == 68 and nei_node.x == 22 and nei_node.y == 8 and global_new_t == 3:
            #     print(f"🔍 Agent 68 trying to move to (22,8) at time 3!")
            #     if vc_hard_np is not None:
            #         constraint_value = vc_hard_np[22, 8, 3]
            #         print(f"  Constraint value at (22,8) time 3: {constraint_value}")
            #         if constraint_value != 0:
            #             print(f"  ❌ SHOULD BE BLOCKED by agent {constraint_value}")
            #         else:
            #             print(f"  ✅ No constraint found - this is the BUG!")

            # 🚀 Create state ID for fast lookup
            nei_state_id = (nei_node.id << 16) | global_new_t

            # 🚀 Fast lookup using sets (O(1) instead of O(n))
            if nei_state_id in open_set_ids or nei_state_id in closed_set_ids:
                continue

            # # ✅ NEW: If we've reached goal, only allow staying at goal (unless constrained)
            # if goal_reached_at is not None and global_new_t > goal_reached_at:
            #     if nei_node != goal_node:
            #         continue  # Must stay at goal after reaching it

                # #Check if staying at goal is constrained
                # if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                #     # print(
                #     #     f"⚠️ Agent {agent.num if agent else 'X'} cannot stay at goal at time {global_new_t} due to constraints")
                #     continue  # Cannot stay at goal due to constraints

            # 🚀 Optimized constraint checking for normal movement
            if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                continue
            try:
                new_h = nei_node.get_cached_heuristic(goal_node.xy_name, h_dict)
            except:
                goal_h_dict = h_dict[goal_node.xy_name]
                new_h = int(goal_h_dict[nei_node.x, nei_node.y])

            nei_astr_node = AStarNode(nei_node, new_t, new_h, parent=next_astr_node)
            #nei_astr_node = AStarNode(nei_node, global_new_t, new_h, parent=next_astr_node)
            heapq.heappush(open_list, nei_astr_node)
            open_set_ids.add(nei_state_id)
            open_list_names.append(nei_astr_node.xyt_name)

        # Add to closed list
        closed_set_ids.add(next_astr_node.state_id)
        closed_list_names.append(next_astr_node.xyt_name)

    # If we exit the loop, try to return best path found
    # if best_goal_node is not None:
    #     path = reconstruct_path_to_goal_then_stay(best_goal_node, goal_node, path_length,
    #                                               vc_hard_np, ec_hard_np, pc_hard_np, agent)
    #     runtime = time.time() - start_time
    #     return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}

    # Return same format as original
    # if sum(resources.max_nodes) <= 0:
    #     print(f"no more resources")
    runtime = time.time() - start_time
    return None, {'runtime': runtime}

# 🚀 Helper function for fast constraint checking
def is_constrained_fast(current_node: AStarNode, next_node: Node, global_new_t: int,
                        vc_hard_np, ec_hard_np, pc_hard_np) -> bool:
    """Fixed constraint checking with proper bounds checking"""

    # Vertex constraint with bounds checking
    # if (vc_hard_np is not None and
    #         global_new_t < vc_hard_np.shape[2] and
    #         next_node.x < vc_hard_np.shape[0] and
    #         next_node.y < vc_hard_np.shape[1]):
    #
    #     constraint_value = vc_hard_np[next_node.x, next_node.y, global_new_t]
    #     if constraint_value != 0:
    #         return True
    if (vc_hard_np is not None and global_new_t < vc_hard_np.shape[2]):
        constraint_value = vc_hard_np[next_node.x, next_node.y, global_new_t]
        if constraint_value != -1:  # -1 means no constraint
            return True

    # Edge constraints with bounds checking
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

    # Permanent constraint
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
    # First, get path to goal
    path: List[Node] = []
    node_current = goal_node
    while node_current is not None:
        path.append(node_current.n)
        node_current = node_current.parent
    path.reverse()

    goal_reached_at = len(path) - 1

    # Then extend by staying at goal, checking constraints
    while len(path) < target_length:
        next_time = len(path)

        # Check if staying at goal is constrained at this time
        if (vc_hard_np is not None and
                next_time < vc_hard_np.shape[2] and
                vc_hard_np[goal_position.x, goal_position.y, next_time] != 0 and
                vc_hard_np[goal_position.x, goal_position.y, next_time] != (agent.num if agent else -1)):
            #(f"⚠️ Agent {agent.num if agent else 'X'} blocked from staying at goal at time {next_time}")
            break  # Cannot extend further due to constraints

        path.append(goal_position)

    return path


def check_future_goal_constraints(goal_node, current_time, path_length, step_iter, vc_hard_np, pc_hard_np, agent):
    """Check if there are any future constraints at the goal position"""
    future_constraints = []

    # Check from current_time + 1 to path_length
    for local_t in range(current_time + 1, path_length):
        global_t = local_t + step_iter

        # Check vertex constraint
        if (vc_hard_np is not None and
                global_t < vc_hard_np.shape[2] and
                vc_hard_np[goal_node.x, goal_node.y, global_t] != -1):
            future_constraints.append(local_t)

    return future_constraints


def construct_goal_filled_path(goal_node_astar, goal_node, path_length):
    """Construct path to goal, then fill remaining with goal node"""
    # Reconstruct path to goal
    path = []
    node_current = goal_node_astar
    while node_current is not None:
        path.append(node_current.n)
        node_current = node_current.parent
    path.reverse()

    # Fill remaining path with goal node
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
        vc_soft_np: np.ndarray | None,  # Optional soft constraints (currently unused)
        ec_soft_np: np.ndarray | None,
        pc_soft_np: np.ndarray | None,
        resources,  # The class that contains the limit of the nodes
        max_depth: int = 500,  # Limit on the depth of the search
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent=None,
        agent_index: int = 0,  # Current agent index
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

    # 🚀 Use sets for faster lookups, but keep string names as backup
    open_set_ids: Set[int] = {start_astr_node.state_id}  # Fast integer lookup
    open_list_names: List[str] = [start_astr_node.xyt_name]  # Backward compatibility
    closed_set_ids: Set[int] = set()  # Fast integer lookup
    closed_list_names: List[str] = []  # Backward compatibility

    # 🚀 Pre-compute constraint bounds (optimization)
    max_vc_time = vc_hard_np.shape[-1] if vc_hard_np is not None else 0
    max_ec_time = ec_hard_np.shape[-1] if ec_hard_np is not None else 0
    max_constraint_time = min(max_vc_time, max_ec_time) if max_ec_time > 0 else max_vc_time

    # Track when goal is reached
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

        # ✅ NEW: If we've reached the desired path length, construct and return path
        # if next_astr_node.t >= path_length:
        #     if best_goal_node is not None:
        #         # Construct path: reach goal + stay at goal
        #         path = reconstruct_path_to_goal_then_stay(best_goal_node, goal_node, path_length,
        #                                                   vc_hard_np, ec_hard_np, pc_hard_np, agent)
        #         runtime = time.time() - start_time
        #         return path, {'runtime': runtime, 'open_list': open_list, 'closed_list': closed_list_names}
        #     else:
        #         # Haven't reached goal yet, continue searching
        #         pass

        # Explore neighbors
        for nei_node in next_astr_node.neighbours_nodes:
            new_t = next_astr_node.t + 1
            global_new_t = new_t + step_iter
            # In the neighbor exploration loop:


            if global_new_t > path_length or global_new_t >= max_constraint_time:
                continue

            # 🚀 Create state ID for fast lookup
            nei_state_id = (nei_node.id << 16) | global_new_t

            # 🚀 Fast lookup using sets (O(1) instead of O(n))
            if nei_state_id in open_set_ids or nei_state_id in closed_set_ids:
                continue

            # ✅ NEW: If we've reached goal, only allow staying at goal (unless constrained)
            if goal_reached_at is not None and global_new_t > goal_reached_at:
                if nei_node != goal_node:
                    continue  # Must stay at goal after reaching it

                #Check if staying at goal is constrained
                if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                    # print(
                    #     f"⚠️ Agent {agent.num if agent else 'X'} cannot stay at goal at time {global_new_t} due to constraints")
                    continue  # Cannot stay at goal due to constraints

            # 🚀 Optimized constraint checking for normal movement
            if is_constrained_fast(next_astr_node, nei_node, global_new_t, vc_hard_np, ec_hard_np, pc_hard_np):
                # if agent and agent.num == 68 and nei_node.x == 22 and nei_node.y == 8 and global_new_t == 3:
                #     print(f"  ✅ is_constrained_fast correctly blocked the move")
                continue
            # if agent and agent.num == 68 and nei_node.x == 22 and nei_node.y == 8 and global_new_t == 3:
            #     print(f"  ❌ BUG: is_constrained_fast did NOT block the move!")

            # 🚀 Use cached heuristic
            try:
                new_h = nei_node.get_cached_heuristic(goal_node.xy_name, h_dict)
            except:
                goal_h_dict = h_dict[goal_node.xy_name]
                new_h = int(goal_h_dict[nei_node.x, nei_node.y])

            nei_astr_node = AStarNode(nei_node, new_t, new_h, parent=next_astr_node)

            heapq.heappush(open_list, nei_astr_node)
            open_set_ids.add(nei_state_id)
            open_list_names.append(nei_astr_node.xyt_name)

        # Add to closed list
        closed_set_ids.add(next_astr_node.state_id)
        closed_list_names.append(next_astr_node.xyt_name)

    runtime = time.time() - start_time
    return None, {'runtime': runtime}


def recursive_construct_path(goal_node: AStarNode, goal_position: Node, target_length: int,
                                       vc_hard_np, ec_hard_np, pc_hard_np, agent,         start_node: Node,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        resources,  # The class that contains the limit of the nodes
        max_depth: int = 500,  # Limit on the depth of the search
        max_final_time: int = int(1e10),
        flag_k_limit: bool = False,
        k_limit: int = int(1e10),
        inf_num: int = int(1e10),
        agent_index: int = 0,  # Current agent index
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

    # Then extend by staying at goal, checking constraints
    while len(path) < target_length:
        next_time = len(path)
        next_node = goal_position
        # Check if staying at goal is constrained at this time
        if (vc_hard_np is not None and
                next_time < vc_hard_np.shape[2] and
                vc_hard_np[goal_position.x, goal_position.y, next_time] != 0 and
                vc_hard_np[goal_position.x, goal_position.y, next_time] != (agent.num if agent else -1)):
            #(f"⚠️ Agent {agent.num if agent else 'X'} blocked from staying at goal at time {next_time}")
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
