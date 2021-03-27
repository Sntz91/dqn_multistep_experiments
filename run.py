import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import collections
import matplotlib.pyplot as plt
import time
import argparse

from tensorboardX import SummaryWriter

from lib import wrappers 
#from lib.dueling_dqn import DuelingDQN as DQN
from lib.dqn import DQN
#from lib.replay_memory import ReplayMemory
from lib.multistep_replay_memory import Experience
from lib.multistep_replay_memory import MultiStepBuffer

#ENV_NAME = "BreakoutDeterministic-v4"
ENV_NAME = "PongNoFrameskip-v4"
GAMMA = 0.99
BATCH_SIZE = 32

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_LAST_FRAME = 200000

TARGET_UPDATE = 10000
MEMORY_SIZE = 100000 #1000000
MEMORY_START_SIZE = 10000
LEARNING_RATE = 0.00001
#N_FRAMES = 4000000
N_FRAMES = 1500000
N_STEP = 1


class Agent():
    def __init__(self, env, replay_memory):
        self.env = env
        self.replay_memory = replay_memory
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon, device):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            current_state = np.array([self.state], copy=False)
            state_v = torch.tensor(current_state).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        next_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         done, next_state)
        self.replay_memory.append(exp)
        self.state = next_state

        if done:
            done_reward = self.total_reward
            self._reset()
        return done_reward 


def calc_loss(batch, net, tgt_net, device):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.from_numpy(states).to(device)
    next_states_v = torch.from_numpy(next_states).to(device)
    actions_v = torch.from_numpy(actions).to(device)
    rewards_v = torch.from_numpy(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    #.unsqueeze(-1) makes array of array
    #.squeeze(-1) makes array again
    # give value of said actions
    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        #get q-value of target net
        next_state_values = tgt_net(next_states_v).max(1)[0]
        #every state_value where state is terminal -> 0
        next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach() #remove comp graph
    expected_state_action_values = (next_state_values * GAMMA**args.nstep) + rewards_v
    
    return F.smooth_l1_loss(state_action_values,
                            expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nstep", type=int, default=N_STEP, help="n-step size for replay memory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Start playing on device: ", device)
    env = wrappers.make_env(ENV_NAME)
    #env = gym.wrappers.Monitor(env, "video", force=True)
    policy_net = DQN(env.observation_space.shape,
                     env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape,
                     env.action_space.n).to(device)
    writer = SummaryWriter(comment="-N_STEP_"+str(args.nstep)+"-" + ENV_NAME)
    
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = EPSILON_START
    #rm = ReplayMemory(MEMORY_SIZE)
    rm = MultiStepBuffer(MEMORY_SIZE, args.nstep, GAMMA)
    agent = Agent(env, rm) 
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)    
    total_rewards = []
    best_m_reward = None
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    #reward None until final step in episode
    while True:
        frame_idx += 1

        epsilon = max(EPSILON_END, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)


        reward = agent.play_step(policy_net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            print("%d: done %d games, reward %.3f, m_reward %.3f, "
                  "eps %.2f, speed %.2f f/s" 
                  % (frame_idx, len(total_rewards), reward, m_reward,
                     epsilon, speed)
                )
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(policy_net.state_dict(), 
                           "models/" + ENV_NAME + "/" + str(args.nstep) + "/best_%.0f.dat"
                           % m_reward)
                best_m_reward = m_reward

            if frame_idx >= N_FRAMES:
                break

        if(len(rm) < MEMORY_START_SIZE):
            continue

        if frame_idx % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        optimizer.zero_grad()
        batch = rm.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, policy_net, target_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()

#TODO: Hyperparameter for Pong and Parameter for Breakout
#TODO: Epsilon linearly from 1 to 0.1 in 10^6 frames, then to 0.01 until rest
#TODO: Training to 350 fps!
#TODO: Solve Pong faster! 30MIN!
#TODO: Model will overfit at some time and exploit the AI... solve?
