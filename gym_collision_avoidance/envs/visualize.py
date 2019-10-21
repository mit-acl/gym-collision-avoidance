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

def get_plot_save_dir(plot_save_dir, plot_policy_name):
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
    save_for_animation=False):
    if max([agent.step_num for agent in agents]) == 0:
        return

    plot_save_dir, plot_policy_name, base_fig_name = get_plot_save_dir(plot_save_dir, plot_policy_name)

    fig = plt.figure(env_id, figsize=(10, 8))

    xlim = ylim = None
    if xlim is not None and ylim is not None:
        fig = plt.figure(env_id, figsize=(10, 10))
    else:
        fig = plt.figure(env_id, figsize=(10, 8))

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

    # title_string = "Episode"
    # plt.title(title_string)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')


    # plotting style (only show axis on bottom and left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.draw()

    if xlim is not None and ylim is not None:
        plt.xlim([-5,6])
        plt.ylim([-2,7])
    else:
        ax.axis('equal')

    if in_evaluate_mode:
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

    plt.pause(0.0001)

    # def render(self, mode='human', close=False):
    #     if not Config.ANIMATE_EPISODES:
    #         return
    #     if close:
    #         if self.viewer is not None:
    #             self.viewer.close()
    #             self.viewer = None
    #         return

    #     screen_width = 600
    #     screen_height = 400

    #     world_width = self.max_x - self.min_x
    #     world_height = self.max_y - self.min_y
    #     scale_x = screen_width/world_width
    #     scale_y = screen_height/world_height

    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(screen_width, screen_height)

    #     if self.begin_episode:
    #         self.begin_episode = False
    #         self.goaltrans = []
    #         self.agenttrans = []
    #         self.viewer.geoms = []
    #         '''
    #         xs = np.linspace(self.min_dist_to_goal, self.max_dist_to_goal, 100)
    #         ys = self._height(xs)
    #         xys = list(zip((xs-self.min_dist_to_goal)*scale, ys*scale))

    #         self.track = rendering.make_polyline(xys)
    #         self.track.set_linewidth(4)
    #         self.viewer.add_geom(self.track)

    #         clearance = 10

    #         l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
    #         car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         car.add_attr(rendering.Transform(translation=(0, clearance)))
    #         self.cartrans = rendering.Transform()
    #         car.add_attr(self.cartrans)
    #         self.viewer.add_geom(car)
    #         '''

    #         for i, agent in enumerate(self.agents):
    #             goal_icon = rendering.make_circle(10)
    #             goal_icon.add_attr(rendering.Transform(translation=(0, 10)))
    #             self.goaltrans.append(rendering.Transform())
    #             goal_icon.add_attr(self.goaltrans[i])
    #             goal_icon.set_color(plt_colors[i][0],
    #                                 plt_colors[i][1],
    #                                 plt_colors[i][2])
    #             self.viewer.add_geom(goal_icon)

    #             agent_icon = rendering.make_circle(scale_x*agent.radius)
    #             agent_icon.set_color(plt_colors[i][0],
    #                                  plt_colors[i][1],
    #                                  plt_colors[i][2])
    #             agent_icon.add_attr(rendering.Transform(translation=(0, 0)))
    #             self.agenttrans.append(rendering.Transform())
    #             agent_icon.add_attr(self.agenttrans[i])
    #             self.viewer.add_geom(agent_icon)

    #         # flagx = (agent.dist_to_goal-self.min_position)*scale
    #         # flagy1 = self._height(self.goal_position)*scale
    #         # flagy2 = flagy1 + 50
    #         # flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
    #         # self.viewer.add_geom(flagpole)

    #     #  flag = rendering.FilledPolygon([(flagx, flagy2),
    #     #                                 (flagx, flagy2-10),
    #     #                                 (flagx+25, flagy2-5)])
    #     #     flag.set_color(.8,.8,0)
    #     #     self.viewer.add_geom(flag)

    #     else:
    #         for i, agent in enumerate(self.agents):
    #             self.goaltrans[i].set_translation(
    #                 (agent.goal_global_frame[0] - self.min_x) * scale_x,
    #                 (agent.goal_global_frame[1] - self.min_y) * scale_y)
    #             self.agenttrans[i].set_translation(
    #                 (agent.pos_global_frame[0] - self.min_x) * scale_x,
    #                 (agent.pos_global_frame[1] - self.min_y) * scale_y)

    #             agent_traj = rendering.make_circle(1)
    #             agent_traj.add_attr(rendering.Transform(translation=(0, 0)))
    #             agent_traj.set_color(plt_colors[i][0],
    #                                  plt_colors[i][1],
    #                                  plt_colors[i][2])
    #             agenttrans = rendering.Transform()
    #             agent_traj.add_attr(agenttrans)
    #             agenttrans.set_translation(
    #                 (agent.pos_global_frame[0] - self.min_x) * scale_x,
    #                 (agent.pos_global_frame[1] - self.min_y) * scale_y)
    #             self.viewer.add_geom(agent_traj)

    #     rgb_array = mode == 'rgb_array'
    #     return self.viewer.render(return_rgb_array=rgb_array)