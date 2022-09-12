import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_pointer = 0
        self.mem_state = np.zeros((self.mem_size, input_shape))
        self.mem_state_new = np.zeros((self.mem_size, input_shape))
        self.mem_action = np.zeros((self.mem_size, n_actions))
        self.mem_reward = np.zeros((self.mem_size))
        self.mem_terminal = np.zeros((self.mem_size), dtype=bool)

    def store_trans (self, state, action, reward, state_, done):
        index = self.mem_pointer % self.mem_size
        self.mem_state[index] = state
        self.mem_action[index] = action
        self.mem_reward[index] = reward
        self.mem_state_new[index] = state_
        self.mem_terminal[index] = done
        self.mem_pointer += 1

    def sample_buffer (self, batch_size):
        max_mem = min(self.mem_pointer, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        state = self.mem_state[batch]
        state_ = self.mem_state_new[batch]
        action = self.mem_action[batch]
        reward = self.mem_reward[batch]
        done = self.mem_terminal[batch]

        return state, action, reward, state_, done


    def print_mem(self):
        print(self.mem_state)

rb = ReplayBuffer(100, 4, 2)
rb.print_mem()