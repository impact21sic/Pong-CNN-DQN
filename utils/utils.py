## Imports
import torch.nn as nn
from collections import deque, namedtuple
import numpy as np
import random
import torch
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


## Replay Buffer
class ReplayBuffer():

    def __init__(self, batch_size, capacity=20_000):
        """
        Replay buffer that holds examples in the form of (s, a, r, s').

        args:
            capacity (int): the size of the buffer, default 20000
            batch_size (int): size of batches used for sampling
        """
        self.batch_size = batch_size
        self.capacity = capacity
        # the buffer holds all (s, a, r, s') pairs
        # and pops the last set when the capacity is reached
        self.buffer = deque(maxlen=self.capacity)
        # the examples in the buffer are saved as transitions
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, state, action, reward, next_state, done):
        """
        Places an (s, a, r, s') example converted into type torch.Tensor in the buffer.

        args:
            state (np.array): the observation obtained from the environment
            action (list[int]): the action taken
            reward (list[int]): the reward received from the action taken at the current state
            next_state (np.array or None): the observation obtained from the environment after 
                                           the action, is None when the state is a terminal state
        """
        # convert all to tensors
        state = torch.FloatTensor(state)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        done = torch.ByteTensor([done])

        if next_state is not None:
            next_state = torch.FloatTensor(next_state)

        # create a transition
        transition = self.Transition(state=state, action=action, reward=reward, next_state=next_state, done=done)
        # append in the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """
        Samples a random transition of size batch_size from the buffer.

        returns:
            transitions (namedtuple): a tuple of form (s, a, r, s', done)
        """
        transitions = random.sample(self.buffer, self.batch_size)

        return self.Transition(*(zip(*transitions)))

    def __len__(self):
        """
        Returns the length of the memory, i.e. number of examples in the memory.

        returns:
            length (int): number of examples in the memory
        """

        return len(self.buffer)


## Neural network
class Net(nn.Module):

    def __init__(self, input_idx, output_idx, img_height=84, img_width=84):
        """Implementing Neural network wiht pytorch nn.Module object"""
        super(Net, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.input_idx = input_idx
        self.output_idx = output_idx
        # conv layers
        self.conv1 = nn.Conv2d(self.input_idx, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        # fully connected layers
        self.fc1 = nn.Linear(self.fc_size(), 512)
        self.fc2 = nn.Linear(512, self.output_idx) 

    def forward(self, x):
        """Implementing nn.Module.forward."""
        x = x.unsqueeze(0)
        x = x.view(-1, self.input_idx, self.img_height, self.img_width)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        self.fc_size()
        # flattens the (N, C, H, W) to (N, C*H*W)
        x = self.relu(self.fc1(x.view(x.size(0), -1)))

        return self.fc2(x)

    def fc_size(self):
        """Calculates the size of the output features from the convolutional layers."""
        x = np.expand_dims(np.zeros((4, 84, 84)), 0)
        x = torch.tensor(x, dtype=torch.float32)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        return x.view(x.size(0), -1).shape[1]

    def step(self, state, epsilon):
        """
        Using epsilon-greedy strategy for choosing an action, this balances exploration
        and exploitation by choosing between both randomly.

        args:
            state (np.array): the observation obtained from the environment
            epsilon (float): epsilon to manage the probability of exploit vs explore
            actions taken in the environment

        returns:
            action (int): action to be taken from the environment
        """
        # pick random action
        if random.random() < epsilon:
            action = random.randrange(self.output_idx)

        else:
            # pick the predicted Q value for a state
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                # get the predicted state-action values
                q_values = self.forward(state) 
                action = q_values.max(1)[1].item()

        return action


## Image preprocessing
def image_preprocess(state):
    """
    Image preprocessing function which resizes and crops the image.
    It is used to simplify the solution the network is working on.

    args:
        state (np.array): the observation obtained from the environment

    returns:
        state (np.array): preprocessed observation i.e. resized and cropped
        for simplicity.
    """
    states = []

    # cropping the unnecessary parts of the image
    # and turning it into grayscale square of 84x84
    for i in range(state.shape[0]):
        states.append(cv2.resize(cv2.cvtColor(state[i][36:185, :, :], cv2.COLOR_RGB2GRAY), 
            (84, 84), interpolation=cv2.INTER_AREA))

    # chaning the values to be in (0, 1)
    states = np.array(states) / 255.0
    
    return states
