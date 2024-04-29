import gym_examples
import gymnasium as gym
import time
import torch
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

class PolicyNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size=256):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, output_size)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        return self.softmax(self.fc2(torch.nn.functional.relu(self.fc1(x))))
        # return self.fc2(torch.nn.functional.relu(self.fc1(x)))

    def get_action_and_logp(self, x):
        action_prob = self.forward(x)
        # action = action_prob.detach().numpy()
        m = torch.distributions.Categorical(action_prob)
        action = m.sample()
        logp = m.log_prob(action)
        return action.item(), logp
        #print("@@@@@", action)
        # return action, 0

    def act(self, x):
        action, _ = self.get_action_and_logp(x)
        return action


class ValueNet(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size=64):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))


def policyGradient(env, max_num_steps=502, gamma=0.98, lr=0.01,
                   num_traj=10, num_iter=200):
    input_size = 25
    output_size = 42
    Trajectory = namedtuple('Trajectory', 'states actions rewards dones logp')

    def collect_trajectory(_):
        state_list = []
        action_list = []
        reward_list = []
        dones_list = []
        logp_list = []
        observation, info = env.reset(seed = _*4 + 1)
        state = state_to_nparray(observation)
        done = False
        steps = 0
        while not done:
            action, logp = policy.get_action_and_logp(state)
            # newobservation, reward, done, _ = env.step(action)
            newobservation, reward, done, truncated, info = env.step(action)
            newstate = state_to_nparray(newobservation)
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            dones_list.append(done)
            logp_list.append(logp)
            steps += 1
            state = newstate
            # state = observation

        traj = Trajectory(states=state_list, actions=action_list,
                          rewards=reward_list, logp=logp_list, dones=dones_list)
        return traj

    def calc_returns(rewards):
        dis_rewards = [gamma**i * r for i, r in enumerate(rewards)]
        return [sum(dis_rewards[i:]) for i in range(len(dis_rewards))]

    policy = PolicyNet(input_size, output_size)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    value = ValueNet(input_size)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=lr)

    mean_return_list = []
    for it in range(num_iter):
        traj_list = [collect_trajectory(_) for _ in range(num_traj)]
        returns = [calc_returns(traj.rewards) for traj in traj_list]

        #====================================#
        # policy gradient with base function #
        #====================================#
        policy_loss_terms = [-1. * traj.logp[j] * (returns[i][j] - value(traj.states[j]))
                             for i, traj in enumerate(traj_list) for j in range(len(traj.actions))]

        #====================================#
        # policy gradient with reward-to-go  #
        #====================================#
        #policy_loss_terms = [-1. * traj.logp[j] * (torch.Tensor([returns[i][j]]))
        #                     for i, traj in enumerate(traj_list) for j in range(len(traj.actions))]

        #====================================#
        # policy gradient                    #
        #====================================#
        #policy_loss_terms = [-1. * traj.logp[j] * (torch.Tensor([returns[i][0]]))
        #                     for i, traj in enumerate(traj_list) for j in range(len(traj.actions))]

        policy_loss = 1. / num_traj * torch.cat(policy_loss_terms).sum()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_loss_terms = [1. / len(traj.actions) * (value(traj.states[j]) - returns[i][j])**2.
                            for i, traj in enumerate(traj_list) for j in range(len(traj.actions))]
        value_loss = 1. / num_traj * torch.cat(value_loss_terms).sum()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        mean_return = 1. / num_traj * \
            sum([traj_returns[0] for traj_returns in returns])
        mean_return_list.append(mean_return)
        if it % 10 == 0:
            print('Iteration {}: Mean Return = {}'.format(it, mean_return))

    return policy, mean_return_list

###########################
##########################


def state_to_nparray(state):
    state_array = None
    for key, value in state.items():
        if state_array is None:
            state_array = value
        else:
            state_array = np.concatenate((state_array, value), dtype=np.float32)

    return state_array


def main():
    env = gym.make('gym_examples/CrowdNav-v0')
    # env.action_space.seed(42)

    """episode = 20000
    episode_num = 1

    observation, info = env.reset(seed=episode_num)

    while True:

        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ## print(observation)

        if (terminated or truncated):
            episode_num += 1
            if (episode_num <= episode):
                observation, info = env.reset(seed=episode_num)
            else:
                break"""


    agent, mean_return_list = policyGradient(env, max_num_steps=100, gamma=1.0, num_traj=5, num_iter=20000)

    plt.plot(mean_return_list)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Return')
    plt.savefig('pg1_returns.png', format='png', dpi=300)
    env.close()

if __name__ == '__main__':
    main()