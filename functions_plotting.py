from globals import *


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# PLOT FUNCS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def get_marker_line(alg_name: str):
    if alg_name in markers_lines_dict:
        return markers_lines_dict[alg_name]
    marker_line = '-'
    if 'PrP' == alg_name:
        marker_line += 'v'
    elif 'LNS2' == alg_name:
        marker_line += 'p'
    elif 'PIBT' == alg_name:
        marker_line += 'h'
    elif 'LaCAM' == alg_name:
        marker_line += '1'
    elif 'LaCAM*' == alg_name:
        marker_line += '2'
    elif alg_name in ['CGA-MAPF', 'MACGA']:
        marker_line += 'X'
    #     marker_line += 'd'
    elif 'CGA(L)' == alg_name:
        marker_line += 'X'
    elif alg_name in ['CGA+PIBT', 'MACGA+PIBT']:
        marker_line += 'd'
    else:
        marker_line += random.choice(markers)
    markers_lines_dict[alg_name] = marker_line
    return marker_line


def get_alg_color(alg_name: str):
    if alg_name in colors_dict:
        return colors_dict[alg_name]
    if 'PrP' == alg_name:
        color = 'green'
    elif 'LNS2' == alg_name:
        color = 'blue'
    elif 'PIBT' == alg_name:
        color = 'salmon'
    elif 'LaCAM' == alg_name:
        color = 'indigo'
    elif 'LaCAM*' == alg_name:
        color = 'plum'
    elif alg_name in ['CGA-MAPF', 'MACGA']:
        color = 'red'
    elif 'CGA(L)' == alg_name:
        color = 'purple'
    elif alg_name in ['CGA+PIBT', 'MACGA+PIBT']:
        color = 'brown'
    else:
        color = random.choice(color_names)
    colors_dict[alg_name] = color
    return color


def get_color(i):
    index_to_pick = i % len(color_names)
    return color_names[index_to_pick]


def set_plot_title(ax, title, size=9):
    ax.set_title(f'{title}', fontweight="bold", size=size)


def set_legend(ax, framealpha=None, size=9):
    to_put_legend = True
    # to_put_legend = False
    if to_put_legend:
        if not framealpha:
            framealpha = 0
        legend_properties = {'weight': 'bold', 'size': size}
        # legend_properties = {}
        if framealpha is not None:
            ax.legend(prop=legend_properties, framealpha=framealpha)
        else:
            ax.legend(prop=legend_properties)


# def plot_step_in_env(ax, info):
#     ax.cla()
#     # nodes = info['nodes']
#     # a_name = info['i_agent'].name if 'i_agent' in info else 'agent_0'
#     img_np = info['img_np']
#     agents = info['agents']
#
#     field = img_np * -1
#     ax.imshow(field, origin='lower', cmap='binary')
#
#     others_y_list, others_x_list, others_cm_list, alpha_list = [], [], [], []
#     for agent in agents:
#         if 'i_agent' in info and info['i_agent'] == agent:
#             continue
#         curr_node = agent.curr_node
#         others_y_list.append(curr_node.y)
#         others_x_list.append(curr_node.x)
#         others_cm_list.append(get_color(agent.num))
#         if agent.curr_node == agent.get_goal_node():
#             alpha_list.append(0.2)
#         else:
#             alpha_list.append(1)
#     ax.scatter(others_y_list, others_x_list, s=100, c='k', alpha=alpha_list)
#     ax.scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list), alpha=alpha_list)
#     # ax.scatter(others_y_list, others_x_list, s=50, c='yellow')
#
#     if 'i_agent' in info:
#         i_agent: AgentAlg = info['i_agent']
#         curr_node = i_agent.curr_node
#         next_goal_node = i_agent.get_goal_node()
#         ax.scatter([curr_node.y], [curr_node.x], s=200, c='k')
#         ax.scatter([curr_node.y], [curr_node.x], s=100, c='r')
#         ax.scatter([next_goal_node.y], [next_goal_node.x], s=400, c='white', marker='X', alpha=0.4)
#         ax.scatter([next_goal_node.y], [next_goal_node.x], s=200, c='red', marker='X', alpha=0.4)
#
#     title_str = 'plot_step_in_env\n'
#     if 'to_title' in info:
#         to_title = info['to_title']
#         title_str += f'{to_title}\n '
#     if 'img_dir' in info:
#         img_dir = info['img_dir']
#         title_str += f'Map: {img_dir[:-4]}\n '
#     if 'i' in info:
#         i = info['i']
#         title_str += f'(iteration: {i + 1})\n'
#     title_str += f'{len(agents)} agents '
#     ax.set_title(title_str)
def plot_step_in_env(ax, info):
    ax.cla()
    img_np = info['img_np']
    agents = info['agents']

    field = img_np * -1
    ax.imshow(field, origin='lower', cmap='binary')

    others_y_list, others_x_list, others_cm_list, alpha_list = [], [], [], []
    for agent in agents:
        if 'i_agent' in info and info['i_agent'] == agent:
            continue  # Skip main agent for now
        curr_node = agent.curr_node
        others_y_list.append(curr_node.y)
        others_x_list.append(curr_node.x)
        others_cm_list.append(get_color(agent.num))
        alpha_list.append(0.2 if agent.curr_node == agent.get_goal_node() else 1)

    ax.scatter(others_y_list, others_x_list, s=100, c='k', alpha=alpha_list)
    ax.scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list), alpha=alpha_list)

    if 'i_agent' in info:
        i_agent = info['i_agent']
        curr_node = i_agent.curr_node
        next_goal_node = i_agent.get_goal_node()

        # Make the main agent larger and bolder
        ax.scatter([curr_node.y], [curr_node.x], s=300, c='gold', edgecolors='black', linewidths=2, marker='o', label='Tracked Agent')
        ax.scatter([curr_node.y], [curr_node.x], s=150, c='red', marker='o')

        # Highlight goal with an extra large marker
        ax.scatter([next_goal_node.y], [next_goal_node.x], s=500, c='white', marker='X', alpha=0.6, edgecolors='black', linewidths=1.5)
        ax.scatter([next_goal_node.y], [next_goal_node.x], s=300, c='red', marker='X', alpha=0.6)

    title_str = 'plot_step_in_env\n'
    if 'to_title' in info:
        title_str += f"{info['to_title']}\n "
    if 'img_dir' in info:
        title_str += f"Map: {info['img_dir'][:-4]}\n "
    if 'i' in info:
        title_str += f"(iteration: {info['i'] + 1})\n"
    title_str += f"{len(agents)} agents "
    ax.set_title(title_str)
    ax.legend()


def plot_sr(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']

    for alg_name in alg_names:
        sr_list = []
        x_list = []
        for n_a in n_agents_list:
            if len(info[alg_name][f'{n_a}']['sr']) > 0:
                sr_list.append(np.sum(info[alg_name][f'{n_a}']['sr']) / len(info[alg_name][f'{n_a}']['sr']))
                x_list.append(n_a)
        ax.plot(x_list, sr_list, get_marker_line(alg_name), color=get_alg_color(alg_name),
                alpha=0.5, label=f'{alg_name}', linewidth=5, markersize=20)
    # ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xlim([min(n_agents_list), max(n_agents_list)])
    ax.set_ylim([0, 1 + 0.1])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', fontsize=27)
    ax.set_ylabel('Success Rate', fontsize=27)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    set_plot_title(ax, f'{img_dir[:-4]}', size=30)
    set_legend(ax, size=24)
    labelsize = 20
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()



def plot_time_metric(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    max_time = info['max_time']

    # x_list = n_agents_list[:4]
    x_list = n_agents_list
    for alg_name in alg_names:
        soc_list = []
        x_list_to_plot = []
        res_str = ''
        for n_a in x_list:
            if len(info[alg_name][f'{n_a}']['time']) > 5:
                x_list_to_plot.append(n_a)
                soc_list.append(np.mean(info[alg_name][f'{n_a}']['time']))
                res_str += f'\t{n_a} - {soc_list[-1]: .2f}, '
        ax.plot(x_list_to_plot, soc_list, get_marker_line(alg_name), color=get_alg_color(alg_name),
                alpha=0.5, label=f'{alg_name}', linewidth=5, markersize=20)
        # print(f'{alg_name}\t\t\t: {res_str}')
    ax.set_xlim([min(x_list) - 20, max(x_list) + 20])
    ax.set_xticks(x_list)
    ax.set_xlabel('N agents', fontsize=27)
    ax.set_ylabel('Runtime', fontsize=27)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]}', size=30)
    labelsize = 20
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    set_legend(ax, size=23)


def plot_time_metric_cactus(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    max_time = info['max_time']

    # x_list = n_agents_list[:4]
    x_list = n_agents_list
    for alg_name in alg_names:
        rt_list = []
        # res_str = ''
        for n_a in x_list:
            rt_list.extend(info[alg_name][f'{n_a}']['time'])
            # res_str += f'\t{n_a} - {rt_list[-1]: .2f}, '
        rt_list.sort()
        ax.plot(rt_list, get_marker_line(alg_name), color=get_alg_color(alg_name),
                alpha=0.5, label=f'{alg_name}', linewidth=2, markersize=10)
        # print(f'{i_alg}\t\t\t: {res_str}')
    # ax.set_xlim([min(x_list) - 20, max(x_list) + 20])
    # ax.set_xticks(x_list)
    ax.set_xlabel('Solved Instances', fontsize=15)
    ax.set_ylabel('Runtime', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {max_time} sec.',
                   size=11)
    # set_legend(ax, size=12)


def plot_soc(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    max_time = info['max_time']
    i_problems = info['i_problems']

    for alg_name in alg_names:
        soc_list = []
        x_list = []
        for n_a in n_agents_list:
            if len(info[alg_name][f'{n_a}']['soc']) < 5:
                break
            x_list.append(n_a)
            soc_list.append(np.mean(info[alg_name][f'{n_a}']['soc']))
        ax.plot(x_list, soc_list, get_marker_line(alg_name), color=get_alg_color(alg_name),
                alpha=0.5, label=f'{alg_name}', linewidth=5, markersize=20)
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', fontsize=27)
    ax.set_ylabel('SoC', fontsize=27)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]}', size=30)
    labelsize = 20
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    # set_legend(ax, size=23)


def plot_soc_cactus(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    max_time = info['max_time']

    for alg_name in alg_names:
        y_list = []
        # res_str = ''
        for n_a in n_agents_list:
            y_list.extend(info[alg_name][f'{n_a}']['soc'])
            # res_str += f'\t{n_a} - {y_list[-1]: .2f}, '
        y_list.sort()
        ax.plot(y_list, get_marker_line(alg_name), color=get_alg_color(alg_name),
                alpha=0.5, label=f'{alg_name}', linewidth=5, markersize=20)
        # print(f'{i_alg}\t\t\t: {res_str}')
    # ax.set_xlim([min(x_list) - 20, max(x_list) + 20])
    # ax.set_xticks(x_list)
    ax.set_xlabel('Solved Instances', fontsize=27)
    ax.set_ylabel('SoC', fontsize=27)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]}',
                   size=30)
    labelsize = 20
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    # set_legend(ax, size=23)


def plot_makespan(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    max_time = info['max_time']

    for alg_name in alg_names:
        makespan_list = []
        x_list_to_plot = []
        for n_a in n_agents_list:
            if len(info[alg_name][f'{n_a}']['makespan']) > 5:
                x_list_to_plot.append(n_a)
                makespan_list.append(np.mean(info[alg_name][f'{n_a}']['makespan']))
        ax.plot(x_list_to_plot, makespan_list, get_marker_line(alg_name), color=get_alg_color(alg_name),
                alpha=0.5, label=f'{alg_name}', linewidth=10, markersize=20)
    # ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    # ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', fontsize=27)
    ax.set_ylabel('Makespan', fontsize=27)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]}', size=30)
    labelsize = 20
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    # set_legend(ax, size=23)


def plot_makespan_cactus(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    max_time = info['max_time']

    for alg_name in alg_names:
        y_list = []
        # res_str = ''
        for n_a in n_agents_list:
            y_list.extend(info[alg_name][f'{n_a}']['makespan'])
            # res_str += f'\t{n_a} - {y_list[-1]: .2f}, '
        y_list.sort()
        ax.plot(y_list, get_marker_line(alg_name), color=get_alg_color(alg_name),
                alpha=0.5, label=f'{alg_name}', linewidth=15, markersize=15)
        # print(f'{i_alg}\t\t\t: {res_str}')
    # ax.set_xlim([min(x_list) - 20, max(x_list) + 20])
    # ax.set_xticks(x_list)
    ax.set_xlabel('Solved Instances', fontsize=27)
    ax.set_ylabel('Makespan', fontsize=27)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]}', size=30)
    labelsize = 20
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    # set_legend(ax, size=23)


def plot_throughput(ax, info):
    ax.cla()
    alg_names = info['alg_names']
    n_agents_list = info['n_agents_list']
    img_dir = info['img_dir']
    n_steps = info['n_steps']

    for alg_name in alg_names:
        throughput_list = []
        for n_a in n_agents_list:
            throughput_list.append(np.mean(info[alg_name][f'{n_a}']['throughput']))
        ax.plot(n_agents_list, throughput_list, get_marker_line(alg_name), color=get_alg_color(alg_name),
                alpha=0.5, label=f'{alg_name}', linewidth=5, markersize=20)
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', fontsize=15)
    ax.set_ylabel('Throughput', fontsize=15)
    # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    set_plot_title(ax, f'{img_dir[:-4]} Map | n_steps: {n_steps}', size=10)
    set_legend(ax, size=12)
