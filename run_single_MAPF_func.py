from globals import *
from functions_general import *
from functions_plotting import *
import json
from valid_check import *


def run_mapf_alg(alg, params, final_render: bool = False):
    set_seed(random_seed_bool=False, seed=123)




    img_dir = params['map']
    n_agents = params['number_of_agents']
    path_to_maps: str = '../maps'
    path_to_heuristics: str = '../logs_for_heuristics'
    path_to_sv_maps: str = '../logs_for_freedom_maps'

    global_start_time = time.time()

    img_np, (height, width) = get_np_from_dot_map(img_dir, path_to_maps)
    map_dim = (height, width)
    graph_start = time.time()
    nodes, nodes_dict = build_graph_from_np(img_np, show_map=False)
    h_dict_start = time.time()
    h_dict: Dict[str, np.ndarray] = exctract_h_dict(img_dir, path_to_heuristics)


    start_nodes, goal_nodes = making_start_and_goal_lists(nodes, params)


    params['img_np'] = img_np
    paths_dict, info = alg(
        start_nodes, goal_nodes, nodes, nodes_dict, h_dict, map_dim, params
    )
    global_runtime = time.time() - global_start_time
    info.update({'map': params['map'],'alg_name': params['alg_name'],
                 'distribution_function': params['distribution_function'],
                 'number_of_agents_as_given': params['number_of_agents'],
                'resources_per_agent': params['resources_per_agent'],
                 'number_of_agents': len(start_nodes), 'run_time': global_runtime,'prefix': params['k_limit']})
    if params['to_save'] == True:
        if 'LNS2-PIBT' in params['alg_name'] or 'PrP-PIBT' in params['alg_name']:
            save_test_LNS2(info,params)
        elif 'PIE' in params['alg_name']:
            save_test_PIE(info,params)
        else:
            save_test(info,params)








