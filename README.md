## Implementing a Deep Reinforcement Learning Model for Autonomous Driving


* Create a python environment for the project. 
* I'd call it **venv** and type out the following command `python -m venv venv`

* Activate the environment `source venv/Script/activate`

* Install all the dependencies in the requirement file `pip install -r requirement.txt`

* Download the Carla version 0.9.08 on windows. 
  Note: This code is only meant to work on windows' OS.

* To run it use `python carla_driver.py --exp-name=ddqn/sac/ppo` command. Don't forget to start the Carla server beforehand.

### Following commands are super useful:
1. --track can be used for wandb
2. tensorboard --logdir runs
3. wandb sync "runs/ddqn_carla-v0.9.08"