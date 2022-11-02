import numpy as np
from parameters import BATCH_SIZE



class ReplayBuffer(object):
    def __init__(self, max_size, img_shape, nav_shape, n_actions):

        self.n_actions = n_actions
        self.buffer_size = max_size
        self.counter = 0
        self.img_state_memory = np.zeros((self.buffer_size, *img_shape), dtype=np.float32)
        self.new_img_state_memory = np.zeros((self.buffer_size, *img_shape), dtype=np.float32)
        self.nav_data = np.zeros((self.buffer_size, *nav_shape), dtype=np.float32)
        self.new_nav_data = np.zeros((self.buffer_size, *nav_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.buffer_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.buffer_size, dtype=np.bool)

    def save_transition(self, state, action, nav_data, reward, new_state, new_nav_data, done):

        index = self.counter % self.buffer_size
        self.img_state_memory[index] = state
        self.new_img_state_memory[index] = new_state
        self.action_memory[index] = action
        self.nav_data[index] = nav_data.reshape(4,1)
        self.new_nav_data[index] = new_nav_data.reshape(4,1)
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.counter += 1

    def sample_buffer(self):
        max = min(self.counter, self.buffer_size)
        batch = np.random.choice(max, BATCH_SIZE, replace=True)

        states = self.img_state_memory[batch]
        nav_data = self.nav_data[batch]
        new_state = self.new_img_state_memory[batch]
        new_nav_data = self.new_nav_data[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self. terminal_memory[batch]

        return states, actions, nav_data, rewards, new_state, new_nav_data, dones

    def __len__(self):
        return len(self.buffer_size)
