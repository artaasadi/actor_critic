import os
from turtle import forward
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distribution.normal import Normal

class CriticNetwork(nn.Module):
    def __init__(self, beta, n_actions, input_dims, name='critic', fc1_dims= 256, fc2_dims= 256, chkp_dir='model'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkp_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"sac_model")

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims) # this critic agent is critcizing both state and action
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr= beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        value = self.fc1(T.cat([state, action], dim=1))
        value = T.relu(value)
        value = self.fc2(value)
        value = T.relu(value)
        value = self.q(value)

        return value

    def save_checkpoint(self):
        print("____saving checkpoint for critic____")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("____loading checkpoint for critic____")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Model):
    def __init__(self, beta, input_dims, name='actor', fc1_dims= 256, fc2_dims= 256, chkp_dir='model'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_dir = chkp_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        