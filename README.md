### Implementing a Deep Reinforcement Learning Model for Autonomous Driving

python carla_driver.py --exp-name=ddqn/sac/ppo 
--track can be used for wandb
tensorboard --logdir runs
wandb sync "runs/ddqn_carla-v0.9.08"