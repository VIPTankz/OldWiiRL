import gym
from gym.wrappers import FrameStack
import numpy as np
from DDDQN import Agent
#from utils import plotLearning
import time
from MarioKartEnv import MarioKartEnv

if __name__ == '__main__':
    env = MarioKartEnv()
    env = FrameStack(env,4)
    save_interval = 400
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1, batch_size=32, n_actions=4,
                      eps_end=0.1, input_dims=[4,32,64], lr=6.25e-5,
                      max_mem_size=1000000,memory = "PER",image = True,
                      learning_starts=25000,replace=12000,preprocess = True,
                      n_step = 4,noisy = True,action_repeat=1)


    #learning starts to 50k
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
        observation = np.stack( observation, axis=0)
        
        score = 0
        i += 1 

        while not done:
            steps += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = np.stack( observation_, axis=0)
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

    
