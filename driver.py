from SimulationClient.connection import ClientConnection
from parameters import EPISODES
from SimulationClient.environment import CarlaEnvironment
from Network.agent import Agent
import sys
import random
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
import pickle
from threading import Thread


logging.basicConfig(filename='client.log', level=logging.DEBUG,
                    format='%(levelname)s:%(message)s')

'''
def plot_graph(x, y, xlabel, ylabel, filename):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, label='DDQN', color='purple')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
'''

def runner():

    np.random.seed(42)
    torch.manual_seed(42)

    checkpoint_load = True

    n_actions = 18  # Car can only make 12 actions
    agent = Agent(n_actions)
    train_thread = Thread(target=agent.train_step, daemon=True)
    train_thread.start()

    avg_score = None
    scores = list()

    fieldnames = ['Epoch', 'Epsilon', 'Reward', 'Average_Reward']
    epoch = 1

    if checkpoint_load:
        agent.load_model()
        with open('serialized_data.pickle', 'rb') as f:
            data = pickle.load(f)
            epoch = data['epoch']
            avg_score = data['avg_score']
            #agent.replay_buffer = data['replay_buffer']
            agent.epsilon = data['epsilon']

    try:
        client, world = ClientConnection()._setup()
        settings = world.get_settings()
        settings.no_rendering_mode = True
        world.apply_settings(settings)

        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError

    env = CarlaEnvironment(client, world)


    while agent.replay_buffer.counter < agent.replay_buffer.buffer_size:
        visual_obs, nav_data = env._reset()
        done = False
        while not done:
            action = random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
            new_visual_obs, new_nav_data, reward, done, _ = env._step(action)
            agent.save_transition(visual_obs, action, nav_data, reward, new_visual_obs, new_nav_data, int(done))
            visual_obs = new_visual_obs
            nav_data = new_nav_data
    
    print('Memory has been initialized!!!\n')

    try:
        for i in range(epoch+1, EPISODES+1):

            print('\nStarting Game: ', i, '\tEpsilon now: {}'.format(agent.epsilon))

            done = False
            visual_obs, nav_data = env._reset()
            score = 0

            while not done:
                action = agent.pick_action(visual_obs, nav_data)

                new_visual_obs, new_nav_data, reward, done, _ = env._step(action)

                score += reward
                agent.save_transition(visual_obs, action, nav_data, reward, new_visual_obs, new_nav_data, int(done))
                agent.train()

                visual_obs = new_visual_obs
                nav_data = new_nav_data
            logging.debug("Done True.")
            scores.append(score)
            if checkpoint_load:
                avg_score = ((avg_score * epoch) + score) / (epoch+1)
            else:
                avg_score = np.mean(scores)


            print('Score:\t\t%.2f ' % score, '\tAverage Score: %.2f' % avg_score, '\tEpsilon: %.2f' % agent.epsilon)

            if i >= 20 and i % 20 == 0:

                agent.save_model()

                with open('learning_data.csv', 'a') as file_handle:
                    writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
                    info = {'Epoch': int(i), 'Epsilon': round(agent.epsilon, 3), 'Reward': round(score,2), 'Average_Reward': round(avg_score,2)}
                    writer.writerow(info)

                data_obj = {'avg_score': avg_score, 'epsilon': agent.epsilon,'epoch': i}#,'replay_buffer': agent.replay_buffer}
                with open('serialized_data.pickle', 'wb') as handle:
                    pickle.dump(data_obj, handle)

        #ep_num = [i+1 for i in range(EPISODES)]
        #plot_graph(ep_num, avg_scores, 'Training Epochs', 'Average Return', 'episodic_average_reward.png')
        train_thread.join()

    finally:
        logging.info("Exiting.")


if __name__ == "__main__":
    try:
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
