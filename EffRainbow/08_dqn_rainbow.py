#!/usr/bin/env python3
import gym
import ptan_actions
import ptan_agent
import ptan_experience
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim

from tensorboardX import SummaryWriter

import networks
from lib import dqn_model, common
from DolphinEnvVec import DolphinEnvVec
import keyboard
# n-step
REWARD_STEPS = 3

# priority replay
PRIO_REPLAY_ALPHA = 0.5
BETA_START = 0.4
BETA_FRAMES = 20000000

# C51
Vmax = 6
Vmin = -2
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            dqn_model.NoisyLinear(512, n_actions)
        )
        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            dqn_model.NoisyLinear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))

if __name__ == "__main__":
    params = common.HYPERPARAMS['MarioBros']
    #params['epsilon_frames'] *= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    
    args = parser.parse_args()
    #print(args.cuda)
    device = torch.device("cuda")# if args.cuda else "cpu"

    with open('pid_num.txt', 'w') as f:
        f.write(str(0))

    #env = DolphinEnv()
    #env = wrap_env(env)

    vec_envs = DolphinEnvVec(4)

    #print("Vector Obs Shape: " + str(vec_envs.observation_space))

    #envs.observation_space = gym.spaces.Box(
            #low=0, high=1, shape=(3,78, 94), dtype=np.uint8)

    #env = DolphinEnv(pid = 0) #gym.make(params['env_name'])
    #env = wrap_env(env,3)

    print(vec_envs.action_space)
    print(vec_envs.observation_space.shape)
    #env = ptan.common.wrappers.wrap_dqn_custom(env)

    #Test this code:
    #check to see if observations are uints or floats

    #raise Exception("stop")

    #need to copy and reshape network

    writer = SummaryWriter(comment="-" + params['run_name'] + "-rainbow")
    #net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    #net = networks.ImpalaCNNSmall(env.observation_space.shape[0], env.action_space.n).to(device)
    net = networks.ImpalaCNNLarge(vec_envs.observation_space.shape[0],vec_envs.action_space.n).to(device)
    #net.load_checkpoint()
    
    tgt_net = ptan_agent.TargetNet(net)
    selector = ptan_actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'],eps_dec=params['epsilon_dec'],eps_min=params['epsilon_final'])
    #ptan_actions.StickyEpsilonGreedyActionSelector()
    agent = ptan_agent.DQNAgent(net, selector, device=device)

    exp_source = ptan_experience.ExperienceSourceFirstLast(vec_envs, agent, gamma=params['gamma'], steps_count=REWARD_STEPS,vectorized=True)#
    buffer = ptan_experience.PrioritizedReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'], eps=0.0025/params['batch_size'])

    frame_idx = 0
    beta = BETA_START

    save_interval = 320000
    start_timer = time.time()

    scores = []
    run_name = "ResultsItems.npy"

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            
            frame_idx += 8
            buffer.populate(8)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx):
                    break

            if frame_idx % 1600 == 0:
                print("Total FPS: " + str(round(frame_idx / (time.time() - start_timer),2)))

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)

            loss_v, sample_prios_v = common.calc_loss_dqn(batch, batch_weights, net, tgt_net.target_model,
                                params['gamma'] ** REWARD_STEPS, device=device)

            """loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                               params['gamma'] ** REWARD_STEPS, device=device)"""

            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % save_interval == 0:
                net.save_checkpoint()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
                np.save(run_name, reward_tracker.get_scores())


