"""
# 학습 코드

## env.step 결과
observation:
  Robot: array(5)
    distance
    v_pref
    velocity_x
    velocity_y
    radius

  human1~5: array(7)
    distance
    position_vector_x
    position_vector_y
    velocity_vector_x
    velocity_vector_y
    radius
    human_radius + robot_radius

action: array(2)
  v_pref
  angle

info:
  distance: float

"""
import collections
import random

import argparse
import numpy as np

import gym_examples_v2
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

learning_rate = 0.005
gamma = 0.99
buffer_limit = 1000 * 5  # size of replay buffer
batch_size = 32
action_count = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  # double-ended queue

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).to(device), \
               torch.tensor(a_lst).to(device), \
               torch.tensor(r_lst, dtype=torch.float).to(device), \
               torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
               torch.tensor(done_mask_lst).to(device)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_count)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon=0):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            random.randint(0, action_count - 1)
        else:
            return out.argmax().item()


class DuelingQnet(nn.Module):
    def __init__(self):
        super(DuelingQnet, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, action_count)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = F.relu(self.fc_value(x))
        a = F.relu(self.fc_adv(x))
        v = self.value(v)
        a = self.adv(a)
        a_avg = torch.mean(a)
        q = v + a - a_avg
        return q

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, action_count - 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)

        # DQN
        # max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        # Double DQN
        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
        max_q_prime = q_target(s_prime).gather(1, argmax_Q)

        target = r + gamma * max_q_prime * done_mask

        # MSE Loss
        loss = F.mse_loss(q_a, target)

        # Smooth L1 Loss
        # loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def state_to_nparray(state):
    return state['Robot'][0:2]
    state_array = None
    for key, value in state.items():
        if state_array is None:
            state_array = value
        else:
            state_array = np.concatenate((state_array, value), dtype=np.float32)

    return state_array


def main(args):
    if args.mode == 'test':
        env = gym.make('gym_examples_v2/CrowdNav-v0', render_mode='human', action_count=action_count)

        q = torch.load('dqn_q.pth', map_location='cpu')
        # q = DuelingQnet().to(device)

        for n_epi in range(1000):
            epsilon = 0
            state, info = env.reset(seed=1)
            state = state_to_nparray(state)
            steps = 0
            total_reward = 0

            done = False

            while not done:
                action = q.sample_action(torch.tensor(state), epsilon=epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = state_to_nparray(next_state)
                steps += 1
                total_reward += reward

                if terminated or truncated:
                    done = True

                if done:
                    break

                if steps % 10 == 0:
                    print(f'steps: {steps}, avg_reward: {total_reward / steps:.4f}, total_reward: {total_reward:.4f}')

                state = next_state

            if n_epi % 10 == 0:
                print(f'episode: {n_epi}')

        return

    env = gym.make('gym_examples_v2/CrowdNav-v0', render_mode='rgb_array', action_count=action_count)
    env.action_space.seed(42)

    episode = 20000
    total_reward = 0
    print_interval = 10
    total_steps = 0
    max_reward = -100
    min_reward = 100000
    max_avg_reward = 0

    q = DuelingQnet().to(device)
    q_target = DuelingQnet().to(device)
    # q = Qnet().to(device)
    # q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for episode_num in range(1, episode + 1):
        state, info = env.reset(seed=1)
        state = state_to_nparray(state)
        done = False
        steps = 0
        episode_reward = 0

        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_mask_list = []

        while not done:
            epsilon = max(0.01, 0.15 - 0.01 * (episode_num / 200))  # Linear annealing from 8% to 1%
            action = q.sample_action(torch.tensor(state).to(device), epsilon=epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = state_to_nparray(next_state)
            total_reward += reward
            episode_reward += reward

            env_state = info['env_state']

            if terminated or truncated:
                done = True

            done_mask = 0.0 if done else 1.0

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            done_mask_list.append(done_mask)

            state = next_state

            if done:
                if env_state == 'success' or env_state == 'timeout':
                    for state, action, reward, next_state, done_mask in zip(state_list, action_list, reward_list,
                                                                            next_state_list, done_mask_list):
                        memory.put((state, action, reward, next_state, done_mask))
                break

            steps += 1
            total_steps += 1

        if memory.size() > buffer_limit * 0.3:
            train(q, q_target, memory, optimizer)

        if max_reward < episode_reward:
            print(f'save checkpoint - max reward updated: {episode_reward:.4f}')
            max_reward = episode_reward
            torch.save(q, 'dqn_q.pth')
            torch.save(q_target, 'dqn_target.pth')

        if min_reward > episode_reward:
            print(f'min reward updated: {episode_reward:.4f}')
            min_reward = episode_reward

        if episode_num % print_interval == 0:
            avg_reward = total_reward / print_interval

            # if max_avg_reward < avg_reward:
            #     print(f'save checkpoint - max avg. reward updated: {avg_reward:.4f}')
            #     max_avg_reward = avg_reward
            #     torch.save(q, 'dqn_q.pth')
            #     torch.save(q_target, 'dqn_target.pth')

            q_target.load_state_dict(q.state_dict())
            print(
                f'n_episode: {episode_num}, buffer: {memory.size()}, avg_steps: {total_steps / print_interval}, avg_reward: {avg_reward:.4f}, eps: {epsilon * 100:.2f}')
            total_reward = 0
            total_steps = 0

    env.close()


if __name__ == '__main__':
    print(f'torch seed: {torch.seed()}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    args = parser.parse_args()

    main(args)
