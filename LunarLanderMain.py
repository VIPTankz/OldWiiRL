import gym
import numpy as np
from DDDQN import Agent
#from utils import plotLearning
import time

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_games = 250
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=0.1, batch_size=32, n_actions=4,
                      eps_end=0.1, input_dims=[8], lr=0.001,
                      max_mem_size=1000000,memory = "PER")

    if load_checkpoint:
        agent.load_models()

    filename = 'LunarLander-Dueling-DDQN-512-Adam-lr0005-replace100.png'
    scores = []
    eps_history = []
    n_steps = 0
    start = time.time()
    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        if i % 20 == 0:
            print('episode: ', i,'score %.1f ' % score,
                 ' average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_games)]
    #plotLearning(x, scores, eps_history, filename)
    print("Total Wall Time: " + str(time.time() - start))
    print(avg_score)
    
