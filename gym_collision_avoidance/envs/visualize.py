import numpy as np
from gym_collision_avoidance.envs.util import find_nearest, rgba2rgb
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.patches as ptch
import glob
import imageio

import moviepy.editor as mp

matplotlib.rcParams.update({'font.size': 24})

plt_colors = []
plt_colors.append([0.8500, 0.3250, 0.0980])  # red
plt_colors.append([0.0, 0.4470, 0.7410])  # blue
plt_colors.append([0.4660, 0.6740, 0.1880])  # green
plt_colors.append([0.4940, 0.1840, 0.5560])  # purple
plt_colors.append([0.9290, 0.6940, 0.1250])  # orange
plt_colors.append([0.3010, 0.7450, 0.9330])  # cyan
plt_colors.append([0.6350, 0.0780, 0.1840])  # chocolate

def get_plot_save_dir(plot_save_dir, plot_policy_name, agents=None):
    if plot_save_dir is None:
        plot_save_dir = os.path.dirname(os.path.realpath(__file__)) + '/../logs/test_cases/'
        os.makedirs(plot_save_dir, exist_ok=True)
    if plot_policy_name is None:
        plot_policy_name = agents[0].policy.str

    base_fig_name = "{test_case}_{policy}_{num_agents}agents{step}.{extension}"
    return plot_save_dir, plot_policy_name, base_fig_name

def animate_episode(num_agents, plot_save_dir=None, plot_policy_name=None, test_case_index=0):
    plot_save_dir, plot_policy_name, base_fig_name = get_plot_save_dir(plot_save_dir, plot_policy_name)
    
    # Load all images of the current episode (each animation)
    fig_name = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = num_agents,
            test_case = str(test_case_index).zfill(3),
            step="_*",
            extension='png')
    last_fig_name = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = num_agents,
            test_case = str(test_case_index).zfill(3),
            step="",
            extension='png')
    all_filenames = plot_save_dir+fig_name
    last_filename = plot_save_dir+last_fig_name

    # Dump all those images into a gif (sorted by timestep)
    filenames = glob.glob(all_filenames)
    filenames.sort()
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        os.remove(filename)
    for i in range(10):
        images.append(imageio.imread(last_filename))

    # Save the gif in a new animations sub-folder
    animation_filename = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = num_agents,
            test_case = str(test_case_index).zfill(3),
            step="",
            extension='gif')
    animation_save_dir = plot_save_dir+"animations/"
    os.makedirs(animation_save_dir, exist_ok=True)
    animation_filename = animation_save_dir+animation_filename
    imageio.mimsave(animation_filename, images)

    # convert .gif to .mp4
    clip = mp.VideoFileClip(animation_filename)
    clip.write_videofile(animation_filename[:-4]+".mp4")

def plot_episode(agents, in_evaluate_mode,
    env_map=None, test_case_index=0, env_id=0,
    circles_along_traj=True, plot_save_dir=None, plot_policy_name=None,
    save_for_animation=False, limits=None, fig_size=(10,8), show=False, save=False):
    if max([agent.step_num for agent in agents]) == 0:
        return

    plot_save_dir, plot_policy_name, base_fig_name = get_plot_save_dir(plot_save_dir, plot_policy_name)

    fig = plt.figure(env_id)
    fig.set_size_inches(fig_size[0], fig_size[1])

    plt.clf()

    ax = fig.add_subplot(1, 1, 1)

    # if env_map is not None:
    #     ax.imshow(env_map.static_map, extent=[-env_map.x_width/2., env_map.x_width/2., -env_map.y_width/2., env_map.y_width/2.], cmap=plt.cm.binary)

    max_time = max([agent.global_state_history[agent.step_num-1, 0] for agent in agents] + [1e-4])
    max_time_alpha_scalar = 1.2
    for i, agent in enumerate(agents):

        # Plot line through agent trajectory
        color_ind = i % len(plt_colors)
        plt_color = plt_colors[color_ind]

        if circles_along_traj:
            plt.plot(agent.global_state_history[:agent.step_num, 1],
                     agent.global_state_history[:agent.step_num, 2],
                     color=plt_color, ls='-', linewidth=2)
            plt.plot(agent.global_state_history[0, 3],
                     agent.global_state_history[0, 4],
                     color=plt_color, marker='*', markersize=20)
            
            # Display circle at agent pos every circle_spacing (nom 1.5 sec)
            circle_spacing = 0.4
            circle_times = np.arange(0.0, agent.global_state_history[agent.step_num-1, 0],
                                     circle_spacing)
            _, circle_inds = find_nearest(agent.global_state_history[:agent.step_num, 0],
                                          circle_times)
            for ind in circle_inds:
                alpha = 1 - \
                        agent.global_state_history[ind, 0] / \
                        (max_time_alpha_scalar*max_time)
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                             radius=agent.radius, fc=c, ec=plt_color,
                             fill=True))

            # Display text of current timestamp every text_spacing (nom 1.5 sec)
            text_spacing = 1.5
            text_times = np.arange(0.0, agent.global_state_history[agent.step_num-1, 0],
                                   text_spacing)
            _, text_inds = find_nearest(agent.global_state_history[:agent.step_num, 0],
                                        text_times)
            for ind in text_inds:
                y_text_offset = 0.1
                alpha = agent.global_state_history[ind, 0] / \
                    (max_time_alpha_scalar*max_time)
                if alpha < 0.5:
                    alpha = 0.3
                else:
                    alpha = 0.9
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.text(agent.global_state_history[ind, 1]-0.15,
                        agent.global_state_history[ind, 2]+y_text_offset,
                        '%.1f' % agent.global_state_history[ind, 0], color=c)
            # Also display circle at agent position at end of trajectory
            ind = agent.step_num - 1
            alpha = 1 - \
                agent.global_state_history[ind, 0] / \
                (max_time_alpha_scalar*max_time)
            c = rgba2rgb(plt_color+[float(alpha)])
            ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                         radius=agent.radius, fc=c, ec=plt_color))
            y_text_offset = 0.1
            ax.text(agent.global_state_history[ind, 1] - 0.15,
                    agent.global_state_history[ind, 2] + y_text_offset,
                    '%.1f' % agent.global_state_history[ind, 0],
                    color=plt_color)

            # if hasattr(agent.policy, 'deltaPos'):
            #     arrow_start = agent.global_state_history[ind, 1:3]
            #     arrow_end = agent.global_state_history[ind, 1:3] + (1.0/0.1)*agent.policy.deltaPos
            #     style="Simple,head_width=10,head_length=20"
            #     ax.add_patch(ptch.FancyArrowPatch(arrow_start, arrow_end, arrowstyle=style, color='black'))

        else:
            colors = np.zeros((agent.step_num, 4))
            colors[:,:3] = plt_color
            colors[:, 3] = np.linspace(0.2, 1., agent.step_num)
            colors = rgba2rgb(colors)

            plt.scatter(agent.global_state_history[:agent.step_num, 1],
                     agent.global_state_history[:agent.step_num, 2],
                     color=colors)

            # Also display circle at agent position at end of trajectory
            ind = agent.step_num - 1
            alpha = 0.7
            c = rgba2rgb(plt_color+[float(alpha)])
            ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                         radius=agent.radius, fc=c, ec=plt_color))
            # y_text_offset = 0.1
            # ax.text(agent.global_state_history[ind, 1] - 0.15,
            #         agent.global_state_history[ind, 2] + y_text_offset,
            #         '%.1f' % agent.global_state_history[ind, 0],
            #         color=plt_color)

    # Label the axes
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    # plotting style (only show axis on bottom and left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.draw()

    if limits is not None:
        xlim, ylim = limits
        plt.xlim(xlim)
        plt.ylim(ylim)
    else:
        ax.axis('equal')

    if in_evaluate_mode and save:
        fig_name = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = len(agents),
            test_case = str(test_case_index).zfill(3),
            step="",
            extension='png')
        filename = plot_save_dir+fig_name
        plt.savefig(filename)

    if save_for_animation:
        fig_name = base_fig_name.format(
            policy=plot_policy_name,
            num_agents = len(agents),
            test_case = str(test_case_index).zfill(3),
            step="_"+"{:06.1f}".format(max_time),
            extension='png')
        filename = plot_save_dir+fig_name
        plt.savefig(filename)

    if show:
        plt.pause(0.0001)