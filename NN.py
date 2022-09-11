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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+"sac_model")

        self.fc1 = nn.Linear(self.input_dims[0]+self.n_actions, self.fc1_dims) # this critic agent is critcizing both state and action
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


class ValueNetwork(nn.Model):
    def __init__(self, beta, input_dims, name='value', fc1_dims= 256, fc2_dims= 256, chkp_dir='model'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_dir = chkp_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr= beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.fc1(state)
        value = T.relu(value)
        value = self.fc2(state)
        value = T.relu(value)
        value = self.v(value)

        return value

    def save_checkpoint(self):
        print("____saving checkpoint for critic____")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("____loading checkpoint for critic____")
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Model):
    def __init__(self, beta, input_dims, max_actions, fc1_dims= 256, fc2_dims= 256, n_actions= 2, name= 'actor', chkp_dir='model'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.max_actions = max_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.reparam_noise = 1e-6

        self.checkpoint_dir = chkp_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr= beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.fc1(state)
        value = T.relu(value)
        value = self.fc2(value)
        value = T.relu(value)

        mu = self.mu(value)
        sigma = T.clamp(self.sigma(value), min= self.reparam_noise, max= 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize= True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_actions).to(self.device)

        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        print("____saving checkpoint for critic____")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("____loading checkpoint for critic____")
        self.load_state_dict(T.load(self.checkpoint_file))
