import os
from sys import maxsize
import torch as T
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import ReplayBuffer
from NN import CriticNetwork, ValueNetwork, ActorNetwork

class Agent():
    def __init__(self, env=None, input_dims=[8], gamma=0.99, beta= 0.0003, tau= 0.005, n_actions=2, max_size=1000000,
                 fc1_dims= 256, fc2_dims= 256, batch_size = 256, reward_scale= 2,):
        self.gamma = gamma
        self.beta = beta
        self.tau = tau
        self.max_size = max_size
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.reward_scale = reward_scale
        self.memory = ReplayBuffer(self.max_size, self.input_dims, self.n_actions)
        self.env = env
        self.batch_size = batch_size

        self.actor = ActorNetwork(beta= self.beta, input_dims= self.input_dims, max_actions= self.env.action_space.high,
                                    fc1_dims= fc1_dims, fc2_dims= fc2_dims, n_actions= self.n_actions, name= 'actor', chkp_dir= 'model')
        self.critic1 = CriticNetwork(beta= self.beta, n_actions= self.n_actions, input_dims= self.input_dims, name= 'critic1')
        self.critic2 = CriticNetwork(beta= self.beta, n_actions= self.n_actions, input_dims= self.input_dims, name= 'critic2')
        self.value = ValueNetwork(beta= self.beta, input_dims=self.input_dims, name= 'value')
        self.target_value = ValueNetwork(beta= self.beta, input_dims= self.input_dims, name= 'target_value')
        
        self.update_network_parameters(tau=1)
        
    def act(self, observations):
        state = T.tensor([observations]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state= state, reparameterize= False)

        return actions.cpu().detach().numpy()[0]

    def store(self, state, action, reward, state_, done):
        self.memory.store_trans(state= state, action= action, reward= reward, state_= state_, done= done)

    def update_network_parameters(self, tau):
        if tau == None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_params_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_params_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print("____saving models____")
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        print("____loading models____")
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

    def learn(self):
        if self.memory.mem_pointer < self.batch_size:
            return
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype= T.float).to(self.actor.device)
        action = T.tensor(action, dtype= T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype= T.float).to(self.actor.device)
        state_ = T.tensor(state_, dtype= T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state= state, reparameterize= False)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic1.forward(state= state, action= actions)
        q2_new_policy = self.critic2.forward(state= state, action= actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value_optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph= True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state= state, reparameterize= True)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic1.forward(state= state, action= actions)
        q2_new_policy = self.critic2.forward(state= state, action= actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph= True)
        self.actor.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.reward_scale*reward + self.gamma * value_
        q1_old_policy = self.critic1.forward(state, action).view(-1)
        q2_old_policy = self.critic2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_network_parameters()