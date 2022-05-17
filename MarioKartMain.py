import gym
import numpy as np
from DDDQN import Agent
#from utils import plotLearning
import time
from MarioKartEnv import MarioKartEnv

if __name__ == '__main__':
    env = MarioKartEnv()
    save_interval = 250
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1, batch_size=64, n_actions=4,
                      eps_end=0.01, input_dims=[4,32,64], lr=6.25e-5,
                      max_mem_size=1000000,memory = "PER",image = True,
                      learning_starts=25000,replace=12000,preprocess = True,
                      eps_dec=1.5e-6)

    if load_checkpoint:
        agent.load_models()

    scores = []
    steps = 0
    start = time.time()
    i = -1
    arr = []
    while True:
        done = False
        observation = env.reset()
        
        score = 0
        i += 1 

        while not done:
            steps += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_

        arr.append([score,i,steps,round(time.time() - start,4),agent.epsilon])
        if i % save_interval == save_interval - 1:
            np.save("Results.npy",np.array(arr))
            agent.save_models()

        #eps_history.append(agent.epsilon)

    
