import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from threading import Thread
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from parameters import *
from simulation.environment import CarlaEnvironment
from networks.off_policy.ddqn.agent import DQNAgent


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla-0.9.8', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=EPISODES, help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by deafult')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help='if toggled, experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default='autonomous driving', help="wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default="idreesrazak", help="enitity (team) of wandb's project")
    args = parser.parse_args()
    
    return args


def runner():

    #========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    #========================================================================
    
    args = parse_args()
    exp_name = args.exp_name
    try:

        if exp_name == 'ddqn':
            run_name = f"ddqn_{args.env_name}"
        elif exp_name == 'ppo':
            run_name = f"ppo_{args.env_name}"

    except Exception as e:
        print(e.message)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        wandb.tensorboard.patch(root_logdir="runs/{run_name}", save=False, tensorboard_x=True, pytorch=True)
    
    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    
    #========================================================================
    #                           INITIALIZING THE NETWORK
    #========================================================================

    checkpoint_load = MODEL_LOAD

    if exp_name == 'ddqn':
        n_actions = 11  # Car can only make 11 actions
        agent = DQNAgent(n_actions)
    elif exp_name == 'ppo':
        n_actions = 2  # Car can only make 2 actions
        agent = PPOAgent(n_actions)
        #learn_iters = 0
        #n_steps = 0
        #N = 20
    
    #train_thread = Thread(target=agent.train, daemon=True)
    #train_thread.start()

    cumulative_score = None
    scores = list()
    episodic_length = 0
    epoch = 0

    if checkpoint_load:
        agent.load_model()
        if exp_name == 'ddqn':
            with open('checkpoint_ddqn.pickle', 'rb') as f:
                data = pickle.load(f)
                epoch = data['epoch']
                cumulative_score = data['cumulative_score']
                agent.epsilon = data['epsilon']
        elif exp_name == 'ppo':
            with open('checkpoint_ppo.pickle', 'rb') as f:
                data = pickle.load(f)
                epoch = data['epoch']
                cumulative_score = data['cumulative_score']


    #========================================================================
    #                           CREATING THE SIMULATION
    #========================================================================

    try:
        client, world = ClientConnection().setup()
        #settings = world.get_settings()
        #settings.no_rendering_mode = True
        #world.apply_settings(settings)

        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError

    env = CarlaEnvironment(client, world)

    #========================================================================
    #                           INITIALIZING THE MEMORY
    #========================================================================
    
    if exp_name == 'ddqn':
        while agent.replay_buffer.counter < agent.replay_buffer.buffer_size:
            observation = env.reset()
            done = False
            while not done:
                action = random.randint(0,n_actions-1)
                new_observation, reward, done, _ = env.step(action)
                agent.save_transition(observation, action, reward, new_observation, int(done))
                observation = new_observation
    
    #========================================================================
    #                           ALGORITHM
    #========================================================================


    try:
        time.sleep(1)
        for step in range(epoch+1, EPISODES+1):
            if exp_name == 'ddqn':
                print('Starting Episode: ', step, ', Epsilon Now:  {:.3f}'.format(agent.epsilon), ', ', end="")
            else:
                print('Starting Episode: ', step ,', ', end="")

            #Reset
            done = False
            observation = env.reset()
            score = 0

            #Episode start: timestamp
            t1 = datetime.now()

            while not done:
                if exp_name == 'ppo':
                    action, prob, val = agent.get_action(observation)
                else:
                    action = agent.get_action(observation)

                new_observation, reward, done, _ = env.step(action)
                score += reward
                
                if exp_name == 'ppo':
                    agent.save_transition(observation, action, prob, val, reward, done)
                    agent.learn(total_timesteps=200_000_000)
                else:
                    agent.save_transition(observation, action, reward, new_observation, int(done))
                    agent.learn()

                observation = new_observation

            #Episode end : timestamp
            t2 = datetime.now()
            t3 = t2-t1
            episodic_length += abs(t3.total_seconds())

            logging.info("Done == True.")
            
            scores.append(score)

            if checkpoint_load:
                cumulative_score = ((cumulative_score * (step - 1)) + score) / (step)
            else:
                cumulative_score = np.mean(scores)

            print('Reward:  {:.2f}'.format(score), ', Average Reward:  {:.2f}'.format(cumulative_score))

            if step >= 20 and step % 20 == 0:
                agent.save_model()

                if exp_name == 'ddqn':
                    data_obj = {'cumulative_score': cumulative_score, 'epsilon': agent.epsilon,'epoch': step}
                    with open('checkpoint_DDQN.pickle', 'wb') as handle:
                        pickle.dump(data_obj, handle)
                elif exp_name == 'ppo':
                    data_obj = {'cumulative_score': cumulative_score,'epoch': step}
                    with open('checkpoint_PPO.pickle', 'wb') as handle:
                        pickle.dump(data_obj, handle)

                writer.add_scalar("Reward/info", np.mean(scores[-20:]), step)
                writer.add_scalar("Cumulative Reward/info", cumulative_score, step)
                writer.add_scalar("Episode Length (s)/info", episodic_length/20, step)
                if exp_name == 'ddqn':
                    writer.add_scalar("Epsilon/info", agent.epsilon, step)

                episodic_length = 0

        #train_thread.join()

    finally:
        logging.info("Exiting.")


if __name__ == "__main__":
    try:
        
        logging.basicConfig(filename='client.log', level=logging.DEBUG,format='%(levelname)s:%(message)s')
        runner()

    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
