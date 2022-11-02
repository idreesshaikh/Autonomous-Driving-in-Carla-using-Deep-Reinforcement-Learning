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
LATENT_DIM = 200

#Dueling DQN (hyper)parameters
DQN_LEARNING_RATE = 0.0001
EPSILON = 1.00
EPSILON_END = 0.02
EPSILON_DECREMENT = 0.0000002

REPLACE_NETWORK = 10
DQN_CHECKPOINT_DIR = 'models/ddqn'
MODEL_ONLINE = 'carla_dueling_dqn_online.pth'
MODEL_TARGET = 'carla_dueling_dqn_target.pth'
'''
#SAC (hyper)parameters
SAC_LEARNING_RATE = 3e-4
SAC_CHECKPOINT_DIR = 'models/sac'
TAU = 0.005
ACTOR_SAC = 'carla_actor_sac.pth'
CRITIC_1_SAC = 'carla_critic_1_sac.pth'
CRITIC_2_SAC = 'carla_critic_2_sac.pth'
VALUE_SAC = 'carla_value_sac.pth'
TARGET_VALUE_SAC = 'carla_target_value_sac.pth'
'''
#Proximal Policy Optimization (hyper)parameters
PPO_LEARNING_RATE = 2.5e-4  
PPO_CHECKPOINT_DIR = 'models/ppo'
POLICY_CLIP = 0.2
PPO_ACTOR = 'carla_actor_ppo.pth'
PPO_CRITIC = 'carla_critic_ppo.pth'
PPO_BATCH_SIZE = 5
