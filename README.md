## Implementing a Deep Reinforcement Learning Model for Autonomous Driving


* Create a python environment for the project

* I'd call it **venv** and type out the following command `python -m venv venv`

* Activate the environment `source venv/Script/activate`

* Install all the dependencies in the requirement file `pip install -r requirement.txt`

* Download the Carla version 0.9.08 on windows
  Note: This code is only meant to work on windows' OS.

* To run it use `python continuous_driver.py --exp-name=ppo` command. Don't forget to start the Carla server beforehand.

### Following commands are super useful:

* `--track` can be used for wandb
* tensorboard --logdir runs
* wandb sync "runs/PPO"