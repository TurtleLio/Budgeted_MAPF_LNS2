import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from globals import *


def validate_solution(agents) -> Dict:
    """
    Comprehensive validation of the current solution.

    Args:
        agents: List of AgentAlg objects with .path attribute

    Returns:
        dict: Validation results with detailed information about any issues found
    """
    results = {
        'is_valid': True,
        'path_coherence': {'valid': True, 'issues': []},
        'collisions': {'valid': True, 'issues': []},
        'summary': {
            'total_agents': len(agents),
            'agents_with_paths': 0,
            'coherence_violations': 0,
            'collision_violations': 0
        }
    }

    # Check path coherence
    coherence_results = check_path_coherence(agents)
    results['path_coherence'] = coherence_results

    # Check collisions
    collision_results = check_collisions(agents)
    results['collisions'] = collision_results

    # Update summary
    results['summary']['agents_with_paths'] = len([a for a in agents if a.path])
    results['summary']['coherence_violations'] = len(coherence_results['issues'])
    results['summary']['collision_violations'] = len(collision_results['issues'])

    # Overall validity
    results['is_valid'] = coherence_results['valid'] and collision_results['valid']

    return results


def check_path_coherence(agents) -> Dict:
    """
    Check if all agent paths are coherent (only single steps, no jumping).

    Args:
        agents: List of AgentAlg objects

    Returns:
        dict: Coherence validation results
    """
    results = {
        'valid': True,
        'issues': []
    }

    for agent in agents:
        if not hasattr(agent, 'path') or not agent.path:
            results['issues'].append({
                'agent': agent.num,
                'type': 'missing_path',
                'description': f"Agent {agent.num} has no path"
            })
            continue

        path = agent.path

        # Check each step in the path
        for i in range(1, len(path)):
            prev_node = path[i - 1]
            curr_node = path[i]

            # Calculate Manhattan distance between consecutive nodes
            distance = abs(curr_node.x - prev_node.x) + abs(curr_node.y - prev_node.y)

            # Valid moves: stay in place (distance 0) or move to adjacent cell (distance 1)
            if distance > 1:
                results['issues'].append({
                    'agent': agent.num,
                    'type': 'invalid_step',
                    'step': i,
                    'from': f"{prev_node.x}_{prev_node.y}",
                    'to': f"{curr_node.x}_{curr_node.y}",
                    'distance': distance,
                    'description': f"Agent {agent.num} jumped from {prev_node.xy_name} to {curr_node.xy_name} (distance {distance})"
                })

            # Additional check: ensure the move is to a valid neighbor or staying in place
            if distance == 1:
                # Check if it's a valid adjacent move (not diagonal)
                dx = abs(curr_node.x - prev_node.x)
                dy = abs(curr_node.y - prev_node.y)

                if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
                    results['issues'].append({
                        'agent': agent.num,
                        'type': 'diagonal_move',
                        'step': i,
                        'from': f"{prev_node.x}_{prev_node.y}",
                        'to': f"{curr_node.x}_{curr_node.y}",
                        'description': f"Agent {agent.num} made diagonal move from {prev_node.xy_name} to {curr_node.xy_name}"
                    })

    if results['issues']:
        results['valid'] = False

    return results


def check_collisions(agents) -> Dict:
    """
    Check for various types of collisions between agents.

    Args:
        agents: List of AgentAlg objects

    Returns:
        dict: Collision validation results
    """
    results = {
        'valid': True,
        'issues': []
    }

    # Get the maximum path length to ensure we check all time steps
    max_time = 0
    for agent in agents:
        if agent.path:
            max_time = max(max_time, len(agent.path))

    # Check for collisions at each time step
    for t in range(max_time):
        # Check vertex collisions (multiple agents at same position)
        vertex_collisions = check_vertex_collisions_at_time(agents, t)
        results['issues'].extend(vertex_collisions)

        # Check edge collisions (agents swapping positions)
        if t > 0:
            edge_collisions = check_edge_collisions_at_time(agents, t)
            results['issues'].extend(edge_collisions)

    if results['issues']:
        results['valid'] = False

    return results


def check_vertex_collisions_at_time(agents, time_step) -> List[Dict]:
    """
    Check for vertex collisions at a specific time step.

    Args:
        agents: List of AgentAlg objects
        time_step: Time step to check

    Returns:
        List of collision issues found
    """
    issues = []
    position_to_agents = defaultdict(list)

    # Group agents by their position at this time step
    for agent in agents:
        if agent.path and time_step < len(agent.path):
            node = agent.path[time_step]
            position = (node.x, node.y)
            position_to_agents[position].append(agent)

    # Find positions with multiple agents
    for position, agents_at_position in position_to_agents.items():
        if len(agents_at_position) > 1:
            agent_nums = [agent.num for agent in agents_at_position]
            issues.append({
                'type': 'vertex_collision',
                'time': time_step,
                'position': f"{position[0]}_{position[1]}",
                'agents': agent_nums,
                'description': f"Vertex collision at time {time_step}, position {position[0]}_{position[1]}: agents {agent_nums}"
            })

    return issues


def check_edge_collisions_at_time(agents, time_step) -> List[Dict]:
    """
    Check for edge collisions (swapping) at a specific time step.

    Args:
        agents: List of AgentAlg objects
        time_step: Time step to check

    Returns:
        List of collision issues found
    """
    issues = []

    # Get all agent positions at time_step-1 and time_step
    agent_moves = []
    for agent in agents:
        if (agent.path and
                time_step < len(agent.path) and
                time_step - 1 < len(agent.path)):
            prev_pos = (agent.path[time_step - 1].x, agent.path[time_step - 1].y)
            curr_pos = (agent.path[time_step].x, agent.path[time_step].y)
            agent_moves.append({
                'agent': agent.num,
                'from': prev_pos,
                'to': curr_pos
            })

    # Check for swapping (edge collisions)
    for i, move1 in enumerate(agent_moves):
        for move2 in agent_moves[i + 1:]:
            # Check if two agents are swapping positions
            if (move1['from'] == move2['to'] and
                    move1['to'] == move2['from'] and
                    move1['from'] != move1['to']):  # Exclude staying in place

                issues.append({
                    'type': 'edge_collision',
                    'time': time_step,
                    'agent1': move1['agent'],
                    'agent2': move2['agent'],
                    'position1': f"{move1['from'][0]}_{move1['from'][1]}",
                    'position2': f"{move1['to'][0]}_{move1['to'][1]}",
                    'description': f"Edge collision at time {time_step}: agents {move1['agent']} and {move2['agent']} swapping positions {move1['from']} <-> {move1['to']}"
                })

    return issues


def print_validation_report(validation_results):
    """
    Print a human-readable validation report.

    Args:
        validation_results: Results from validate_solution()
    """
    print("=" * 60)
    print("SOLUTION VALIDATION REPORT")
    print("=" * 60)

    summary = validation_results['summary']
    print(f"Total agents: {summary['total_agents']}")
    print(f"Agents with paths: {summary['agents_with_paths']}")
    print(f"Solution is valid: {'✓ YES' if validation_results['is_valid'] else '✗ NO'}")
    print()

    # Path coherence report
    coherence = validation_results['path_coherence']
    print(f"PATH COHERENCE: {'✓ VALID' if coherence['valid'] else '✗ INVALID'}")
    if coherence['issues']:
        print(f"  Found {len(coherence['issues'])} coherence violations:")
        for issue in coherence['issues'][:5]:  # Show first 5
            print(f"    - {issue['description']}")
        if len(coherence['issues']) > 5:
            print(f"    ... and {len(coherence['issues']) - 5} more")
    print()

    # Collision report
    collisions = validation_results['collisions']
    print(f"COLLISIONS: {'✓ NO COLLISIONS' if collisions['valid'] else '✗ COLLISIONS FOUND'}")
    if collisions['issues']:
        print(f"  Found {len(collisions['issues'])} collision violations:")
        for issue in collisions['issues'][:5]:  # Show first 5
            print(f"    - {issue['description']}")
        if len(collisions['issues']) > 5:
            print(f"    ... and {len(collisions['issues']) - 5} more")

    print("=" * 60)

def validate_graph_connections(nodes, nodes_dict):
    """
    Validate that all graph connections are valid (distance <= 1, no diagonals).

    Args:
        nodes: List of Node objects
        nodes_dict: Dictionary mapping xy_name to Node objects

    Returns:
        dict: Validation results with details of any invalid connections
    """
    results = {
        'valid': True,
        'invalid_connections': [],
        'diagonal_connections': [],
        'self_connections': [],
        'missing_neighbors': [],
        'summary': {}
    }

    for node in nodes:
        print(f"Validating node {node.xy_name} with {len(node.neighbours)} neighbors")

        for neighbor_name in node.neighbours:
            if neighbor_name not in nodes_dict:
                results['missing_neighbors'].append({
                    'node': node.xy_name,
                    'missing_neighbor': neighbor_name
                })
                continue

            neighbor = nodes_dict[neighbor_name]

            # Check if it's a self-connection
            if neighbor_name == node.xy_name:
                results['self_connections'].append({
                    'node': node.xy_name,
                    'description': f"Node {node.xy_name} connected to itself"
                })
                continue

            # Calculate distance
            distance = abs(neighbor.x - node.x) + abs(neighbor.y - node.y)

            if distance > 1:
                results['invalid_connections'].append({
                    'from_node': node.xy_name,
                    'to_node': neighbor.xy_name,
                    'distance': distance,
                    'description': f"Invalid connection: {node.xy_name} -> {neighbor.xy_name} (distance {distance})"
                })
                results['valid'] = False

            elif distance == 1:
                # Check for diagonal connections
                dx = abs(neighbor.x - node.x)
                dy = abs(neighbor.y - node.y)
                if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
                    results['diagonal_connections'].append({
                        'from_node': node.xy_name,
                        'to_node': neighbor.xy_name,
                        'description': f"Diagonal connection: {node.xy_name} -> {neighbor.xy_name}"
                    })
                    results['valid'] = False

    # Create summary
    results['summary'] = {
        'total_nodes': len(nodes),
        'invalid_connections': len(results['invalid_connections']),
        'diagonal_connections': len(results['diagonal_connections']),
        'self_connections': len(results['self_connections']),
        'missing_neighbors': len(results['missing_neighbors'])
    }
    print(results)
    return results


def validate_constraint_arrays(vc_np, ec_np, pc_np):
    """Check if constraint arrays look reasonable"""
    print("=== CONSTRAINT ARRAY VALIDATION ===")
    print(f"VC non-zero entries: {np.count_nonzero(vc_np)}")
    print(f"EC non-zero entries: {np.count_nonzero(ec_np)}")
    print(f"PC non-minus-one entries: {np.count_nonzero(pc_np != -1)}")

    # Check for unreasonable values
    max_agent_id = np.max(vc_np)
    print(f"Max agent ID in VC: {max_agent_id}")

    if max_agent_id > 200:  # Adjust based on your agent count
        print("❌ WARNING: Suspiciously high agent IDs in constraints")


def debug_path_reconstruction(agent, step_iter):  # First 3 agents
    if len(agent.path) > step_iter:
        current_pos = agent.path[step_iter]
        temp_start = agent.temp_path[0] if agent.temp_path else None

        print(f"Agent {agent.num}:")
        print(f"  step_iter: {step_iter}")
        print(f"  Current position: {current_pos.xy_name}")
        print(f"  temp_path starts at: {temp_start.xy_name if temp_start else 'None'}")
        print(f"  Match: {current_pos == temp_start}")

        if current_pos != temp_start:
            print(f"  ❌ MISMATCH! Agent will jump from {current_pos.xy_name} to {temp_start.xy_name}")


# def validate_temp_paths_only(agents, vc_hard_np, ec_hard_np, pc_hard_np):
#     """Validate just the temp_paths of subset agents"""
#
#     # Create temporary constraint arrays with ONLY temp_paths
#     temp_vc = np.zeros_like(vc_hard_np)
#     temp_ec = np.zeros_like(ec_hard_np)
#     temp_pc = np.full_like(pc_hard_np, -1)
#
#     # Add ALL agents' temp_paths to see if they conflict
#     for agent in agents:  # ALL agents, not just subset
#         update_constraints(agent.temp_path, temp_vc, temp_ec, temp_pc, agent.num)
#
#     # Validate this temp solution
#     validation = validate_solution(agents)  # Using temp_paths
#     print(f"temp_paths validation: {validation['is_valid']}")
#
#     if not validation['is_valid']:
#         print("❌ temp_paths themselves have conflicts!")
#         for issue in validation['collisions']['issues'][:5]:
#             print(f"  {issue['description']}")

# Add this right before reverting to temp_paths
def validate_constraints_consistency(agents: List[AgentAlg], vc_np: np.ndarray,
                                     ec_np: np.ndarray, pc_np: np.ndarray, step_name: str = ""):
    """
    Validates that the constraint arrays are consistent with agent paths
    """
    print(f"🔍 Validating constraints at step: {step_name}")

    # Check vertex constraints
    for agent in agents:
        if hasattr(agent, 'path') and agent.path:
            for t, node in enumerate(agent.path):
                if t < vc_np.shape[2]:
                    constraint_value = vc_np[node.x, node.y, t]
                    if constraint_value != 0 and constraint_value != agent.num:
                        print(f"❌ VC CONFLICT: Agent {agent.num} at ({node.x},{node.y}) time {t}, "
                              f"but constraint shows agent {constraint_value}")
                        return False

    # Check for vertex conflicts between agents
    for t in range(vc_np.shape[2]):
        for x in range(vc_np.shape[0]):
            for y in range(vc_np.shape[1]):
                if vc_np[x, y, t] != 0:
                    # Find all agents at this position and time
                    agents_here = []
                    for agent in agents:
                        if (hasattr(agent, 'path') and agent.path and
                                t < len(agent.path) and
                                agent.path[t].x == x and agent.path[t].y == y):
                            agents_here.append(agent.num)

                    if len(agents_here) > 1:
                        print(f"❌ VERTEX CONFLICT: Multiple agents {agents_here} at ({x},{y}) time {t}")
                        return False

    print(f"✅ Constraints validated successfully at step: {step_name}")
    return True


def debug_constraint_state(agents: List[AgentAlg], vc_np: np.ndarray,
                           ec_np: np.ndarray, pc_np: np.ndarray, step_name: str = ""):
    """
    Debug function to show current constraint state
    """
    print(f"\n📊 Constraint State at {step_name}:")

    # Count constraints
    vc_count = np.count_nonzero(vc_np)
    ec_count = np.count_nonzero(ec_np)
    pc_count = np.count_nonzero(pc_np + 1)  # +1 because -1 means no constraint

    print(f"  Vertex constraints: {vc_count}")
    print(f"  Edge constraints: {ec_count}")
    print(f"  Permanent constraints: {pc_count}")

    # Show which agents have tracking info
    tracked_agents = []
    for agent in agents:
        if (hasattr(agent, 'my_vertex_constraints') and
                (agent.my_vertex_constraints or agent.my_edge_constraints or agent.my_permanent_constraint)):
            tracked_agents.append(agent.num)

    print(f"  Agents with tracking info: {tracked_agents}")

    # Check for orphaned constraints (constraints not tracked by any agent)
    all_tracked_positions = set()
    for agent in agents:
        if hasattr(agent, 'my_vertex_constraints'):
            for x, y, t in agent.my_vertex_constraints:
                all_tracked_positions.add((x, y, t))

    orphaned_count = 0
    for t in range(vc_np.shape[2]):
        for x in range(vc_np.shape[0]):
            for y in range(vc_np.shape[1]):
                if vc_np[x, y, t] != 0 and (x, y, t) not in all_tracked_positions:
                    orphaned_count += 1

    if orphaned_count > 0:
        print(f"  ⚠️  WARNING: {orphaned_count} orphaned constraints detected!")

    print()


def debug_agent_68_paths(agent, stage, step_iter=None, new_path=None):
    """Debug Agent 68's paths at different stages"""
    agent_num = 72
    time = 21
    x = 18
    y = 5
    if agent.num != agent_num:
        return

    print(f"\n🔍 AGENT {agent_num} PATH DEBUG - {stage}")
    if step_iter is not None:
        print(f"  step_iter: {step_iter}")

    # Check main path
    if hasattr(agent, 'path') and agent.path:
        print(f"  agent.path length: {len(agent.path)}")
        print(f"  agent.path first 15: {[(n.x, n.y) for n in agent.path[:15]]}")
        if len(agent.path) > time:
            print(f"  agent.path[{time}]: ({agent.path[time].x}, {agent.path[time].y})")
            if agent.path[time].x == x and agent.path[time].y == y:
                print(f"  ❌ agent.path goes through ({x},{y}) at time {time}!")
    else:
        print(f"  agent.path: None or empty")

    # Check temp_path
    if hasattr(agent, 'temp_path') and agent.temp_path:
        print(f"  agent.temp_path length: {len(agent.temp_path)}")
        print(f"  agent.temp_path first 15: {[(n.x, n.y) for n in agent.temp_path[:15]]}")
        if len(agent.temp_path) > time:
            print(f"  agent.temp_path[{time}]: ({agent.temp_path[time].x}, {agent.temp_path[time].y})")
            if agent.temp_path[time].x == x and agent.temp_path[time].y == y:
                print(f"  ❌ agent.temp_path goes through ({x},{y}) at time {time}!")
    else:
        print(f"  agent.temp_path: None or empty")

    # Check k_path
    if hasattr(agent, 'k_path') and agent.k_path:
        print(f"  agent.k_path length: {len(agent.k_path)}")
        print(f"  agent.k_path first 15: {[(n.x, n.y) for n in agent.k_path[:15]]}")
        if len(agent.k_path) > time:
            print(f"  agent.k_path[{time}]: ({agent.k_path[time].x}, {agent.k_path[time].y})")
            if agent.k_path[time].x == x and agent.k_path[time].y == y:
                print(f"  ❌ agent.k_path goes through ({x},{y}) at time {time}!")
    else:
        print(f"  agent.k_path: None or empty")

    # Check new_path if provided
    if new_path is not None:
        if new_path:
            print(f"  new_path length: {len(new_path)}")
            print(f"  new_path first 15: {[(n.x, n.y) for n in new_path[:15]]}")
            if len(new_path) > time:
                print(f"  new_path[12]: ({new_path[time].x}, {new_path[time].y})")
                if new_path[time].x == x and new_path[time].y == y:
                    print(f"  ❌ new_path goes through ({x},{y}) at time {time}!")
        else:
            print(f"  new_path: None")
