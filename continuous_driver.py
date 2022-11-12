import os
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
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import *


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1e7, help='total timesteps of the experiment')
    parser.add_argument('--episode-length', type=int, default=10000, help='max timesteps in an episode')
    parser.add_argument('--train', type=bool, default=True, help='is it training?')
    parser.add_argument('--load-checkpoint', type=bool, default=False, help='resume training?')
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

        if exp_name == 'ppo':
            run_name = f"PPO"
        else:
            sys.exit()

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
            #sync_tensorboard=True,
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

    #train_thread = Thread(target=agent.train, daemon=True)
    #train_thread.start()

    checkpoint_load = args.load_checkpoint

    run_number = 0
    current_num_files = next(os.walk('logs'))[2]
    run_number = len(current_num_files)
    log_file = 'logs/PPO_carla' + "_log_" + str(run_number) + ".csv"
    
    log_freq = 2e4
    save_freq = 4e4
    action_std = 0.6
    action_std_decay_rate = 5e-2
    min_action_std = 1e-1            
    action_std_decay_freq = 5e5
    
    episodic_length = 0
    timestep = 0
    episode = 0
    cumulative_score = 0

    scores = list()
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
    encode = EncodeState(LATENT_DIM)

    #========================================================================
    #                           ALGORITHM
    #========================================================================

    try:
        time.sleep(1)
        
        agent = PPOAgent(action_std)
        if args.train:

            if checkpoint_load:
                agent.load()
                with open('checkpoints/checkpoint_ppo.pickle', 'rb') as f:
                    data = pickle.load(f)
                    episode = data['episode']
                    timestep = data['timestep']
                    cumulative_score = data['cumulative_score']
        
            #model.learn(args.total_timesteps)
            #print("Checkpointing...")

            # track total training time
            log_f = open(log_file,"w+")
            log_f.write('episode,timestep,reward,cumulative reward\n')

            while timestep < args.total_timesteps:
            
                observation = env.reset()
                observation = encode.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(args.episode_length):
                
                    # select action with policy
                    action = agent.get_action(observation)

                    observation, reward, done, _ = env.step(action)
                    observation = encode.process(observation)
                    
                    agent.memory.rewards.append(reward)
                    agent.memory.dones.append(done)
                    
                    timestep +=1
                    current_ep_reward += reward
                    
                    if timestep % (args.episode_length * 2) == 0:
                        agent.learn()
                    
                    if timestep % action_std_decay_freq == 0:
                        agent.decay_action_std(action_std_decay_rate, min_action_std)

                    if timestep % log_freq == 0:
                        log_f.write('{},{},{},{}\n'.format(episode, timestep, current_ep_reward, cumulative_score))
                        log_f.flush()
                    
                    if timestep % save_freq == 0:
                        agent.save()

                        data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep }
                        with open('checkpoints/checkpoint_ppo.pickle', 'wb') as handle:
                            pickle.dump(data_obj, handle)

                        writer.add_scalar("Reward/info", np.mean(scores[-100:]), episode)
                        writer.add_scalar("Episodic Reward/info", current_ep_reward, episode)
                        writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                        writer.add_scalar("Episode Length (s)/info", episodic_length/episode, episode)
                        writer.add_scalar("Cummulative Reward/(t)", cumulative_score, timestep)
                        writer.add_scalar("Reward/(t)", reward, timestep)

                        episodic_length = 0

                    # break; if the episode is over
                    if done:
                        episode += 1
                        t2 = datetime.now()
                        t3 = t2-t1
                        episodic_length += abs(t3.total_seconds())
                        break

                scores.append(current_ep_reward)
                if checkpoint_load:
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                else:
                    cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
            log_f.close()

        else:
            test_timesteps = 50
            test(env,agent,encode,test_timesteps,args.episode_length)
        #train_thread.join()

    finally:
        logging.info("Exiting.")


def test(env, agent, encode, test_episodes, max_episode_len):

    try:
        agent.load()
        print("Checkpoint loading...")

        running_reward = 0

        for ep in range(1, test_episodes+1):
            ep_reward = 0
            state = env.reset()
            state = encode.process(state)
            
            for t in range(1, max_episode_len+1):
                action = agent.get_action(state)
                state, reward, done, _ = env.step(action)
                state = encode.process(state)
                ep_reward += reward
                
                if done:
                    break

            agent.memory.clear()

            running_reward +=  ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
            ep_reward = 0

        avg_reward = running_reward / test_episodes
        avg_reward = round(avg_reward, 2)
        print("average test reward : " + str(avg_reward))
    except:
        print('Exception occurred. Please train your model first...')

if __name__ == "__main__":
    try:
        logging.basicConfig(filename='logs/ppo.log', level=logging.DEBUG,format='%(levelname)s:%(message)s')
        runner()

    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
