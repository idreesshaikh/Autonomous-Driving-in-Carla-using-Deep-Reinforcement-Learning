from SimulationClient.connection import ClientConnection
from parameters import EPISODES
from SimulationClient.environment import CarlaEnvironment
from Network.agent import Agent
import random
import sys
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
import pickle
from threading import Thread

logging.basicConfig(filename='client.log', level=logging.DEBUG,
                    format='%(levelname)s:%(message)s')


def plot_graph(x, y, xlabel, ylabel, filename):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, label='DDQN', color='purple')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)


def runner():

    np.random.seed(42)
    torch.manual_seed(42)

    checkpoint_load = True

    n_actions = 11  # Car can only make 8 actions
    agent = Agent(n_actions)
    train_thread = Thread(target=agent.train_step, daemon=True)
    train_thread.start()

    scores = list()
    avg_scores = list()
    epsilon_history = list()
    fieldnames = ['Epoch', 'Epsilon', 'Reward', 'Average_Reward']
    current_epoch = 1

    if checkpoint_load:
        agent.load_model()
        with open('serialized_data.pickle', 'rb') as f:
            data = pickle.load(f)
            scores = data['scores']
            avg_scores = data['avg_scores']
            epsilon_history = data['epsilon_history']
            current_epoch = data['epoch']
        agent.epsilon = epsilon_history[-1]

    try:
        client, world = ClientConnection()._setup()
        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError

    env = CarlaEnvironment(client, world)

    try:
        for i in range(EPISODES+1):

            print('\nStarting Game: ', i+1,
                  '\tEpsilon now: {}'.format(agent.epsilon))

            epsilon_history.append(agent.epsilon)
            done = False
            visual_obs, raw_data = env._reset()
            score = 0

            while not done:
                action = agent.pick_action(visual_obs, raw_data)
                new_visual_obs, new_raw_data, reward, done, _ = env._step(
                    action)
                score += reward
                agent.save_transition(
                    visual_obs, raw_data, action, reward, new_visual_obs, new_raw_data, int(done))
                agent.train()
                visual_obs = new_visual_obs
                raw_data = new_raw_data

            logging.debug("Done True.")

            scores.append(score)

            avg_score = np.mean(scores)
            avg_scores.append(avg_score)

            with open('learning_data.csv', 'a') as file_handle:
                writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
                info = {'Epoch': i+current_epoch, 'Epsilon': agent.epsilon,
                        'Reward': score, 'Average_Reward': avg_score}
                writer.writerow(info)

            print('      Episode: ', i+1, '\tScore: %.2f' % score,
                  '\tAverage Score: %.2f' % avg_score, '\tEpsilon: %.2f' % agent.epsilon)

            if i > 20 and i % 20 == 0:

                agent.save_model()

                data_obj = {'scores': scores, 'avg_scores': avg_scores,
                            'epsilon_history': epsilon_history, 'epoch': current_epoch}
                with open('serialized_data.pickle', 'wb') as handle:
                    pickle.dump(data_obj, handle)

        ep_num = [i+1 for i in range(EPISODES)]
        plot_graph(ep_num, avg_scores, 'Training Epochs',
                   'Average Return', 'episodic_average_reward.png')
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
