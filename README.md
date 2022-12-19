# Implementing a Deep Reinforcement Learning Model for Autonomous Driving

## Setup

* Firstly, create a python environment for the project 💥

* I'd call it **venv** `python -m venv venv`

* Activate the environment `source venv/Script/activate`

* By now I presume you have cloned the repo, if not then you better do it

* Install poetry with pip `pip install poetry`

* Now change directry to our poetry directry with `cd poetry/`

* Inside the poetry directory execute this command `poetry update` to install all the dependencies. Once everything is setup up then we're nearly there 

* Download the Carla version 0.9.08 on windows, and after downloading the server
  Note: This code is only meant to work on windows' OS so be mindful of that

* Once the server is up and running, we can start our client with `python continuous_driver.py --exp-name=ppo` command. Don't forget to start the Carla server beforehand. It starts our our training. Yey!!!

## How our Training looks like.

**Town 2** 🏢

![](gifs/town2-car-turn.gif)

*Don't worry this README file will be updated soon.* 
1. Cool gifs, 
2. graphs,
3. diagrams of Architecture, and 
4. a lot more is on its way.

*Just couple of days more* 🌞