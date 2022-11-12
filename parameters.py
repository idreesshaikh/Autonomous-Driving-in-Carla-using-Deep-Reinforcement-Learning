"""

    All the much needed hyper-parameters needed for the algorithm implementation. 

"""
MODEL_LOAD = False
BATCH_SIZE = 1
IM_WIDTH = 128
IM_HEIGHT = 128
GAMMA = 0.99
MEMORY_SIZE = 10000
EPISODES = 10000
LATENT_DIM = 95

#Dueling DQN (hyper)parameters
DQN_LEARNING_RATE = 0.0001
EPSILON = 1.00
EPSILON_END = 0.05
EPSILON_DECREMENT = 0.000001

REPLACE_NETWORK = 4
DQN_CHECKPOINT_DIR = 'preTrained_models/ddqn'
MODEL_ONLINE = 'carla_dueling_dqn_online.pth'
MODEL_TARGET = 'carla_dueling_dqn_target.pth'


#Proximal Policy Optimization (hyper)parameters
PPO_LEARNING_RATE = 2e-4  
PPO_CHECKPOINT_DIR = 'preTrained_models/ppo'
POLICY_CLIP = 0.1
PPO_ACTOR = 'carla_actor_ppo.pth'
PPO_CRITIC = 'carla_critic_ppo.pth'
PPO_BATCH_SIZE = 10