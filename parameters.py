"""

    All the much needed hyper-parameters needed for the algorithm implementation. 

"""

LEARNING_RATE = 0.0001
BATCH_SIZE = 32
IM_WIDTH = 128
IM_HEIGHT = 128
GAMMA = 0.99
EPSILON = 1.00
EPSILON_END = 0.02
EPSILON_DECREMENT = 0.00002
REPLACE_NETWORK = 30
MEMORY_SIZE = 8000
CHECKPOINT_DIR = 'models'
MODEL_ONLINE = 'carla_ddqn_online.pt'
MODEL_TARGET = 'carla_ddqn_target.pt'
EPISODES = 5001
