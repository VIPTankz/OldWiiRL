import gym
import numpy as np
from DDDQN import Agent
import argparse
#from utils import plotLearning
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-srun', type=int, default=0)
    
    args = parser.parse_args()
    srun = args.srun
    
    env = gym.make('LunarLander-v2')
    num_frames = 180000
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=0.1, batch_size=64, n_actions=4,
                      eps_end=0.1, input_dims=[8], lr=0.001,
                      max_mem_size=1000000,memory = "PER",image = False,
                      learning_starts=64,replace=100,n_step = 4,noisy=False)

    if load_checkpoint:
        agent.load_models()

    filename = 'LunarLander-Dueling-DDQN-512-Adam-lr0005-replace100.png'
    scores = []
    eps_history = []
    n_steps = 0
    start = time.time()
    i = -1
    while n_steps < num_frames:
        i += 1
        done = False
        observation = env.reset()
        score = 0

        while not done:
            n_steps += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            #env.render()
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        if i % 10 == 0:
            print('episode: ', i,'score %.1f ' % score,
                 ' average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

        eps_history.append(agent.epsilon)

    #x = [i+1 for i in range(num_games)]
    #plotLearning(x, scores, eps_history, filename)
    #print("Total Wall Time: " + str(time.time() - start))
    #print(avg_score)
    save_stuff = [time.time() - start,avg_score]
    save_stuff = np.array(save_stuff,dtype = float)
    np.save("results_er" + str(srun) + ".npy", save_stuff)
