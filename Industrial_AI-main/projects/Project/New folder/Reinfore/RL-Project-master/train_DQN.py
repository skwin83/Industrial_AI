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
import os

import shutil

import wandb

import gym_examples
import gymnasium as gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import str2bool

learning_rate = 0.005
gamma = 0.99
buffer_limit = 50000  # size of replay buffer
batch_size = 64
print_interval = 20
episode = 10000

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


class DuelingQnet(nn.Module):
    def __init__(self):
        super(DuelingQnet, self).__init__()
        self.fc1 = nn.Linear(23, 128)
        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, 5)

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
            return random.randint(0, 4)

        else:
            return out.argmax().item()


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(23, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon=1.0):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 4)
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
    state_array = None
    for key, value in state.items():
        if state_array is None:
            state_array = value
        else:
            state_array = np.concatenate((state_array, value), dtype=np.float32)

    return state_array


def main(args):
    if args.mode == 'test':
        env = gym.make('gym_examples/CrowdNav-v0', render_mode='human')

        q = torch.load(os.path.join(args.model_path, args.model_name, args.checkpoint_file), map_location='cpu')

        for n_epi in range(1000):
            epsilon = 0
            state, info = env.reset(seed=args.env_seed)
            state = state_to_nparray(state)
            steps = 0
            interval_reward = 0

            done = False

            while not done:
                action = q.sample_action(torch.tensor(state), epsilon=epsilon)
                next_state, reward, terminated, result, info = env.step(action)
                next_state = state_to_nparray(next_state)
                steps += 1
                interval_reward += reward

                if terminated:
                    done = True

                if done:
                    break

                if steps % 10 == 0:
                    print(f'steps: {steps}, avg_reward: {interval_reward / steps:.4f}, total_reward: {interval_reward:.4f}')

                state = next_state

            if n_epi % 10 == 0:
                print(f'episode: {n_epi}')

        return

    interval_reward = 0
    total_reward = 0
    interval_steps = 0
    max_avg_reward = -1000
    success_count = 0
    total_success_count = 0
    collision_count = 0
    total_collision_count = 0
    timeout_count = 0
    total_timeout_count = 0

    env = gym.make('gym_examples/CrowdNav-v0', render_mode='rgb_array')

    # q = Qnet().to(device)
    # q_target = Qnet().to(device)
    q = DuelingQnet().to(device)
    q_target = DuelingQnet().to(device)
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for episode_num in range(1, episode + 1):
        # epsilon = max(0.1, epsilon * 0.99)
        epsilon = max(0.01, 0.15 - 0.01 * (episode_num / 500))  # Linear annealing
        state, info = env.reset(seed=args.env_seed)
        state = state_to_nparray(state)
        done = False

        while not done:
            action = q.sample_action(torch.from_numpy(state).to(device), epsilon)

            next_state, reward, terminated, result, info = env.step(action)
            next_state = state_to_nparray(next_state)
            interval_reward += reward
            total_reward += reward

            if terminated:
                done = True

            done_mask = 0.0 if done else 1.0
            memory.put((state, action, reward, next_state, done_mask))
            state = next_state

            if done:
                if result == 'success':
                    success_count += 1
                    total_success_count += 1
                elif result == 'collision':
                    collision_count += 1
                    total_collision_count += 1
                elif result == 'timeout':
                    timeout_count += 1
                    total_timeout_count += 1
                break

            interval_steps += 1

        if memory.size() > buffer_limit * 0.3:
            train(q, q_target, memory, optimizer)

        if episode_num % print_interval == 0:
            avg_reward = interval_reward / print_interval
            avg_steps = interval_steps / print_interval
            avg_success = success_count / print_interval
            avg_collision = collision_count / print_interval
            avg_timeout = timeout_count / print_interval

            if max_avg_reward < avg_reward:
                checkpoint_file = f'model_q{str(episode_num)}.pt'
                print(f'save checkpoint - max avg reward updated: {avg_reward:.4f} ({checkpoint_file})')
                max_avg_reward = avg_reward
                torch.save(q, os.path.join(args.model_path, args.model_name, checkpoint_file))

            q_target.load_state_dict(q.state_dict())
            print(
                f'n_episode: {episode_num}, buffer: {memory.size()}, avg_steps: {avg_steps}, avg_reward: {avg_reward:.4f}, eps: {epsilon * 100:.2f}, success_rate: {avg_success * 100:.2f}%')

            if args.wandb:
                wandb.log({'episode': episode_num, 'avg_steps': avg_steps, 'epsilon': epsilon, 'avg_reward': avg_reward,
                           'avg_success': avg_success, 'avg_collision': avg_collision, 'avg_timeout': avg_timeout})

            interval_reward = 0
            interval_steps = 0
            success_count = 0
            collision_count = 0
            timeout_count = 0

    print(f'success rate: {total_success_count / episode * 100:.2f} ({total_success_count}/{episode})')
    print(f'collision rate: {total_collision_count / episode * 100:.2f} ({total_collision_count}/{episode})')
    print(f'timeout rate: {total_timeout_count / episode * 100:.2f} ({total_timeout_count}/{episode})')
    print(f'average return: {total_reward / episode:.2f} ({total_reward:.2f}/{episode})')

    with open(f'result_{args.model_name}.txt', 'a', encoding='utf-8') as f:
        f.write(f'{total_success_count / episode * 100:.2f}\t{total_collision_count / episode * 100:.2f}\t{total_timeout_count / episode * 100:.2f}\t{total_reward / episode:.2f}\n')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--env_seed', default='2', type=int, help='1: no, 2: static, 3: dynamic, 4: mixed')
    parser.add_argument('--model_path', default='./models', type=str, help='model path')
    parser.add_argument('--model_name', default='dueling_dqn', type=str, help='algorithm name')
    parser.add_argument('--checkpoint_file', default='model.pt', type=str, help='saved model file')
    parser.add_argument('--overwrite', default=False, type=str2bool, help='overwrite model path (true/false)')
    parser.add_argument('--wandb', default=False, type=str2bool, help='overwrite model path (true/false)')
    parser.add_argument('--train_iters', default='1', type=int, help='number of train iterations')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="CrowdNav-RL", name='Dueling-DQN')

    if args.mode == 'train':
        save_path = os.path.join(args.model_path, args.model_name)
        if args.overwrite:
            try:
                shutil.rmtree(save_path)
                print(f'Delete saved path: {save_path}')
            except FileNotFoundError:
                pass

        os.makedirs(save_path, exist_ok=False)

    for i in range(args.train_iters):
        main(args)
