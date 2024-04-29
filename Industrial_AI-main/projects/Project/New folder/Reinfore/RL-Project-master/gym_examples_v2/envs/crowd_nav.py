import math

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import itertools
import random
import time
import copy

center = [10, 12]
circle_radius = 4  # the distance between the start point of the robot and the goal point

time_step = 0.25  # the time of one step (seconds)

robot_v_pref = 1  # the max speed of the robot

human_num = 5  # the number of human agents

random_human_state = True  # the velocity and radius of human is random (True) or constant (False)

discomfort_dist = 0.5  # the distance which humans feel discomfortbale
discomfort_penalty_factor = 0.2  # the parameter for the reward function when robot is within the distance which humans feel discomfortbale

time_limit = 25  # time limit for visualization 'test'
step_limit = 100  # step limit for training 'train

simulation_purpose = 'train'  # 'train' or 'test'

if simulation_purpose == 'test':
    visualization = "human"
else:
    visualization = "rgb_array"


class CrowdNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_state = 'running'
    target_theta = 0

    def __init__(self, render_mode=visualization, size=32, action_count=8):
        self.size = size  # The pygame screen rate
        self.window_size = 1024  # The size of the PyGame window

        # Observations are dictionaries with the jointstate of the environment.
        self.observation_space = spaces.Dict(
            {
                "Robot": spaces.Box(low=np.array([0, 0, 0, 0, 0]),
                                    high=np.array([10, 10, 10, 10, 10]),
                                    dtype=np.float32),
                "human1": spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0]),
                                     high=np.array([10, 10, 10, 10, 10, 10, 10]),
                                     dtype=np.float32),
                "human2": spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0]),
                                     high=np.array([10, 10, 10, 10, 10, 10, 10]),
                                     dtype=np.float32),
                "human3": spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0]),
                                     high=np.array([10, 10, 10, 10, 10, 10, 10]),
                                     dtype=np.float32),
                "human4": spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0]),
                                     high=np.array([10, 10, 10, 10, 10, 10, 10]),
                                     dtype=np.float32),
                "human5": spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0]),
                                     high=np.array([10, 10, 10, 10, 10, 10, 10]),
                                     dtype=np.float32),
            }
        )

        # We have continuous actions, [linear velocity, angular velocity]
        # self.action_space = spaces.Box(low=np.array([0, -np.pi / 4]),
        #                                high=np.array([robot_v_pref, np.pi / 4]),
        #                                dtype=np.float32)

        self.action_space = spaces.Discrete(action_count)
        self.rotations = list(np.linspace(-np.pi, np.pi, num=action_count))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "Robot": np.array([
                self.get_distance(self._target_location, self.robot.get_position()),
                self.target_theta,
                self.robot.v_pref,
                self.robot.vx,
                self.robot.vy,
                self.robot.radius
            ], dtype=np.float32),
            "human1": np.array([
                self.get_distance(self.robot.get_position(), self.humans[0].get_position()),
                self.get_human_position_vector(self.humans[0])[0],
                self.get_human_position_vector(self.humans[0])[1],
                self.get_human_velocity_vector(self.humans[0])[0],
                self.get_human_velocity_vector(self.humans[0])[1],
                self.humans[0].radius,
                self.humans[0].radius + self.robot.radius
            ], dtype=np.float32),
            "human2": np.array([
                self.get_distance(self.robot.get_position(), self.humans[1].get_position()),
                self.get_human_position_vector(self.humans[1])[0],
                self.get_human_position_vector(self.humans[1])[1],
                self.get_human_velocity_vector(self.humans[1])[0],
                self.get_human_velocity_vector(self.humans[1])[1],
                self.humans[1].radius,
                self.humans[1].radius + self.robot.radius
            ], dtype=np.float32),
            "human3": np.array([
                self.get_distance(self.robot.get_position(), self.humans[2].get_position()),
                self.get_human_position_vector(self.humans[2])[0],
                self.get_human_position_vector(self.humans[2])[1],
                self.get_human_velocity_vector(self.humans[2])[0],
                self.get_human_velocity_vector(self.humans[2])[1],
                self.humans[2].radius,
                self.humans[2].radius + self.robot.radius
            ], dtype=np.float32),
            "human4": np.array([
                self.get_distance(self.robot.get_position(), self.humans[3].get_position()),
                self.get_human_position_vector(self.humans[3])[0],
                self.get_human_position_vector(self.humans[3])[1],
                self.get_human_velocity_vector(self.humans[3])[0],
                self.get_human_velocity_vector(self.humans[3])[1],
                self.humans[3].radius,
                self.humans[3].radius + self.robot.radius
            ], dtype=np.float32),
            "human5": np.array([
                self.get_distance(self.robot.get_position(), self.humans[4].get_position()),
                self.get_human_position_vector(self.humans[4])[0],
                self.get_human_position_vector(self.humans[4])[1],
                self.get_human_velocity_vector(self.humans[4])[0],
                self.get_human_velocity_vector(self.humans[4])[1],
                self.humans[4].radius,
                self.humans[4].radius + self.robot.radius
            ], dtype=np.float32)
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.robot.get_position() - self._target_location, ord=1
            ),
            "env_state": self.env_state,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.step_num = 0
        self.start = time.time()

        self._target_location = np.array([float(center[0]), float(center[1] + circle_radius)])

        self.robot = Robot()
        self.robot.v_pref = robot_v_pref
        self.robot.set_position(np.array([float(center[0]), float(center[1] - circle_radius)]))
        self.robot.set_goal_position(self._target_location)

        self.initial_dg = np.linalg.norm(self.robot.get_position() - self._target_location)

        self.randomize_attributes = random_human_state

        # There are four types of environment.
        if seed % 4 == 1:
            self.train_sim = "no"  # there is no human. (actually, there are fake humans who are very far from robots.)

        elif seed % 4 == 2:
            self.train_sim = "static"  # humans are static.

        elif seed % 4 == 3:
            self.train_sim = "dynamic"  # humans are dynamic.

        elif seed % 4 == 0:
            self.train_sim = "mixed"  # some humans are static and the others are dynamic.

        if self.train_sim == "no":
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_fake_human())

        elif self.train_sim == "static":
            self.humans = []
            total_num = 0
            group_num = random.randrange(1, 5)
            for i in range(group_num - 1):
                while total_num <= human_num - (group_num - (1 + i)):
                    member_num = random.randrange(1, 6)
                    if total_num + member_num <= human_num - (group_num - (1 + i)):
                        total_num += member_num
                        break
                self.humans += self.generate_group(member_num)
            self.humans += self.generate_group(human_num - total_num)


        elif self.train_sim == "dynamic":
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())

        elif self.train_sim == "mixed":
            self.humans = []
            total_num = 0
            group_num = random.randrange(1, 5)
            for i in range(group_num):
                while total_num < human_num - (group_num - (1 + i)):
                    member_num = random.randrange(1, 6)
                    if total_num + member_num < human_num - (group_num - (1 + i)):
                        total_num += member_num
                        break
                self.humans += self.generate_group(member_num)
            for i in range(human_num - total_num):
                self.humans.append(self.generate_circle_crossing_human())

        else:
            raise ValueError("Rule doesn't exist")

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        action = [0.5, self.rotations[action]]

        prev_dg = np.linalg.norm(self.robot.get_position() - self._target_location)

        if simulation_purpose == 'test':
            time.sleep(0.25)
        else:
            self.step_num += 1

        self.robot.move(action)

        if self.train_sim not in ('no', 'static'):
            for human in self.humans:
                human.move()

        current_dg = np.linalg.norm(self.robot.get_position() - self._target_location)

        robot_position = self.robot.get_position()
        target_position = self._target_location
        x_diff = robot_position[0] - target_position[0]
        y_diff = -(robot_position[1] - target_position[1])
        self.target_theta = math.atan2(y_diff, x_diff)

        terminated = False
        success = False
        timeout = False
        collision = False

        dmin = float("inf")
        for i, human in enumerate(self.humans):
            dist_x = human.px - self.robot.px
            dist_y = human.py - self.robot.py
            dist = np.linalg.norm((dist_x, dist_y))
            # closest distance between boundaries of two agents
            closest_dist = dist - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        self.now = time.time()
        if float(self.now - self.start) >= time_limit and simulation_purpose == 'test':
            timeout = True
        elif self.step_num > step_limit and simulation_purpose == 'train':
            timeout = True

        elif current_dg <= self.robot.radius:
            success = True

        # reward function
        if success:  # success reward
            reward = 10
            terminated = True
            self.env_state = "success"

        elif timeout:  # timeout reward
            reward = 0
            terminated = True
            self.env_state = "timeout"

        elif collision:  # collision reward
            reward = -10
            terminated = True
            self.env_state = "collision"

        elif dmin < discomfort_dist:  # discomfortable distance reward
            reward = (dmin - discomfort_dist) * discomfort_penalty_factor

        else:  # otherwise
            reward = -math.log(current_dg + 1)
            # reward = prev_dg - current_dg
            # reward = 1 - (current_dg / prev_dg)
            # reward = 1 - (current_dg / prev_dg) + (self.initial_dg - current_dg) / self.initial_dg
            # reward = (self.initial_dg - current_dg) / self.initial_dg

            # diff = prev_dg - current_dg
            # if diff >= 0:
            #     if self.initial_dg - current_dg > 0:
            #         reward = (self.initial_dg - current_dg) / self.initial_dg
            #     else:
            #         reward = diff
            # else:
            #     reward = diff

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self._target_location * pix_square_size,
                (20, 20),
            ),
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.robot.get_position()) * pix_square_size,
            self.robot.radius * pix_square_size,
        )

        for human in self.humans:
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (human.get_position()) * pix_square_size,
                human.radius * pix_square_size,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def generate_group(self, member_num):
        while True:
            group = []
            center_px = random.uniform(center[0] - circle_radius, center[0] + circle_radius)
            center_py = random.uniform(center[1] - circle_radius, center[1] + circle_radius)
            angle = np.random.random() * np.pi * 2
            for j in range(member_num):
                human = self.generate_group_member(center_px, center_py, angle, member_num, j, group)
                if human == False:
                    group_made = False
                    break
                else:
                    group.append(human)
                    if len(group) == member_num:
                        group_made = True
                        break
            if group_made:
                return group

    def generate_group_member(self, center_px, center_py, angle, member_num, j, group):
        human = Human()
        if self.randomize_attributes:
            human.random_radius()
            human.v_pref = 0
        j_angle = angle + np.pi * 2 * j / member_num
        px = (0.2 + member_num * 0.1) * np.cos(j_angle) + center_px
        py = (0.2 + member_num * 0.1) * np.sin(j_angle) + center_py
        collide = False
        for agent in [self.robot] + self.humans + group:
            min_dist = human.radius + agent.radius
            if np.linalg.norm((px - agent.px, py - agent.py)) < min_dist or \
                    np.linalg.norm((px - agent.gx, py - agent.gy)) < min_dist:
                collide = True
                break
        if collide:
            return False
        else:
            human.px = px
            human.py = py
            human.set("static")
            return human

    def generate_fake_human(self):
        human = Human()
        human.v_pref = 0
        human.radius = 0.3
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px = (20) * np.cos(angle) + 10
            py = (20) * np.sin(angle) + 12
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius  # + self.discomfort_dist
                if np.linalg.norm((px - agent.px, py - agent.py)) < min_dist or \
                        np.linalg.norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break  # jump out of 'for' loop
            if not collide:
                break  # jump out of 'while' loop
        human.px = px
        human.py = py
        human.set("static")
        return human

    def generate_circle_crossing_human(self):
        human = Human()
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = center[0] + (circle_radius * human.v_pref / self.robot.v_pref) * np.cos(angle) + px_noise
            py = center[1] + (circle_radius * human.v_pref / self.robot.v_pref) * np.sin(angle) + py_noise
            # px = (self.circle_radius) * np.cos(angle) + px_noise
            # py = (self.circle_radius) * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius  # + self.discomfort_dist
                if np.linalg.norm((px - agent.px, py - agent.py)) < min_dist or \
                        np.linalg.norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break  # jump out of 'for' loop
            if not collide:
                break  # jump out of 'while' loop
        human.px = px
        human.py = py
        human.set("circle_crossing")
        return human

    def get_distance(self, agent1, agent2):
        return np.linalg.norm((agent1[0] - agent2[0], agent1[1] - agent2[1]))

    def get_human_position_vector(self, human):
        p_vector_x = human.px - self.robot.px
        p_vector_y = human.py - self.robot.py
        return [p_vector_x, p_vector_y]

    def get_human_velocity_vector(self, human):
        v_vector_x = human.vx - self.robot.vx
        v_vector_y = human.vy - self.robot.vy
        return [v_vector_x, v_vector_y]


class Point(object):
    def __init__(self):
        self.radius = 0.3
        self.px = 0
        self.py = 0
        self.vx = 0
        self.vy = 0
        self.gx = 0
        self.gy = 0
        self.v_pref = 0
        self.theta = 0

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def set_goal_position(self, goal_position):
        self.gx = goal_position[0]
        self.gy = goal_position[1]

    def get_position(self):
        return np.array([self.px, self.py])


class Robot(Point):
    def __init__(self):
        super(Robot, self).__init__()

    def move(self, action):
        # self.theta = (self.theta + action[1]) % (2 * np.pi)
        # theta = np.arctan2(action[1], self.vx)
        self.theta = action[1]
        self.vx = action[0] * np.cos(self.theta)
        self.vy = action[0] * np.sin(self.theta)
        self.px = self.px + (self.vx * time_step)
        self.py = self.py + (self.vy * time_step)


class Human(Point):
    def __init__(self):
        super(Human, self).__init__()
        self.v_pref = 1

    def set(self, moving_way="circle_crossing"):
        if moving_way == "circle_crossing":
            self.gx = 20 - self.px
            self.gy = 24 - self.py
            vector_x = self.gx - self.px
            vector_y = self.gy - self.py
            self.theta = np.arctan2(vector_y, vector_x)

        elif moving_way == "square_crossing":
            self.gx = 20 - self.px
            self.gy = self.py
            vector_x = self.gx - self.px
            vector_y = self.gy - self.py
            self.theta = np.arctan2(vector_y, vector_x)

        elif moving_way == "static":
            self.gx = self.px
            self.gy = self.py

    def sample_random_attributes(self):
        self.v_pref = np.random.uniform(0.5, 1.2)
        self.radius = np.random.uniform(0.1, 0.6)

    def random_radius(self):
        self.radius = np.random.uniform(0.1, 0.6)

    def move(self):
        self.vx = self.v_pref * np.cos(self.theta)
        self.vy = self.v_pref * np.sin(self.theta)
        self.px = self.px + self.vx * time_step
        self.py = self.py + self.vy * time_step
        if np.linalg.norm((self.gx - self.px, self.gy - self.py)) < self.radius:
            self.v_pref = 0
