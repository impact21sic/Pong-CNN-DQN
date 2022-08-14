## Imports
import cv2
import gym
import math
import random
import torch
from gym.wrappers import FrameStack
from utils.wrappers import FireReset, EpisodicLifeEnv, MaxAndSkipEnv, NoopResetEnv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from utils.utils import image_preprocess, ReplayBuffer, Net


## Functions and hyperparameters
gamma = 0.99
episodes = 10_000
batch_size = 32 # 32 is default
tau = 1000
capacity = 10_000
replay = ReplayBuffer(batch_size, capacity)
input_idx = 4 
output_idx = 6
net = Net(input_idx, output_idx)
target = Net(input_idx, output_idx)
target.load_state_dict(net.state_dict())
target.eval()
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.0001) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('PongNoFrameskip-v4', render_mode='human')
env = FrameStack(env, 4) # stack 4 frames together to produce the observation
env = FireReset(env) # press 'FIRE'(start) button at the beginning of every game
env = EpisodicLifeEnv(env) # makes end-of-life == end-of-episode
env = MaxAndSkipEnv(env) # returns only every 'skip'-th frame with max pooling across most recent observation
env = NoopResetEnv(env)
epsilon_start = 1.0
epsilon_final = 0.02
epsilon_decay = 49_000
epsilon = lambda frame: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame / epsilon_decay)
losses = []
rewards = []
episode_durations = []
frames = 0
episode_reward = 0


def plot_durations():
    """
    Plots durations.
    """
    plt.figure(1, figsize=[8, 10])
    plt.clf()
    plt.subplot(2, 1, 1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(durations_t.numpy())
    plt.subplot(2, 1, 2)
    plt.title('Loss Function')
    plt.xlabel('Frame')
    plt.plot(losses)
    # pause a bit so that plots are updated
    plt.savefig('score_and_loss.jpeg')
    plt.pause(0.001)
    

def compute_loss(transitions):
    """
    Updates the network's parameters.

    args:
        transitions (namedtuple): a tuple of form (s, a, r, s', done)
        img_heigth (int): height of the image 
        img_width (int): width of the image

    Algorithm for double DQN:
        - get a batch of transitions
        - get a mask for all terminal states
        - get Q values for next_state from both target and main networks
        - get the actions from the network predicted Q values, not the target
    """
    # prepare batches
    state = torch.cat(transitions.state).to(device)
    action = torch.tensor(transitions.action, dtype=torch.long).unsqueeze(1).to(device)
    reward = torch.tensor(transitions.reward).unsqueeze(1).to(device)
    next_state = torch.cat(transitions.next_state).to(device)
    done_mask = torch.ByteTensor(transitions.done).to(device)
    # get Q(s_t, a_t)
    state_action_vals = net(state).gather(1, action)
    # get V(s_{t+1}) using the target model with 
    # actions taken from main model's V(s_{t+1})
    next_state_actions = net(next_state).max(1)[1]
    next_state_vals = target(next_state).gather(1, next_state_actions.unsqueeze(-1))
    # set V(s_{t+1}) = 0 for every terminal state
    next_state_vals[done_mask.type(torch.bool)] = 0.0
    # compute the max expected future return with the Bellman equation
    # and detach() since we don't want gradients for the target model
    expected_max_return = next_state_vals.detach() * gamma + reward
    # compute MSE loss over state-action values and the expected return
    loss = F.mse_loss(state_action_vals, expected_max_return) # uspqhme s MSE, ne s Huber
    # zero the gradients and perform backpropagation
    optimizer.zero_grad()
    loss.backward()

    # clipping the gradients
    for p in net.parameters():
        p.grad.data.clamp_(-1, 1)

    # perform gradient descent step
    optimizer.step()

    return loss.item()


def train():
    global episode_reward, frames

    for episode in range(1, episodes+1):
        # initialize the environment and observe a state
        state = np.array(env.reset())

        for t in count():
            #env.render()
            frames += 1
            # perform epsilon-greedy action selection and step into the environment
            action = net.step(image_preprocess(state), epsilon(frames))
            # observe a new state
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            replay.push(image_preprocess(state), action, 
                    reward, image_preprocess(next_state), done)
            # move to the next state
            state = next_state
            # monitor
            rewards.append(episode_reward)
            print('Episode: ', episode)
            print('Frame: ', frames)
            print('Score: ', episode_reward)
            print('Reward: ', reward)
            print('epsilon decay: ', epsilon(frames))

            if done:
                # monitor
                episode_durations.append(episode_reward)
                plot_durations()
                episode_reward = 0
                break

            # optimize the main network
            if len(replay) > batch_size:
                transitions = replay.sample()
                loss = compute_loss(transitions)
                torch.save(net, 'cnn_dqn_pong_v4.pt')
                losses.append(loss)

            if t % tau == 0:
                target.load_state_dict(net.state_dict())


if __name__ == "__main__":
    train()
