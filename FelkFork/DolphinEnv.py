from multiprocessing import shared_memory
import numpy as np
import time
import os
from PIL import Image
import gym
import random
from copy import deepcopy
import cv2
import warnings
from Wrappers import wrap_env
import subprocess
os.chdir('/home/tyler/Documents/EfficientZero')

warnings.filterwarnings("ignore")

"""
This program implements the standard gym MDP

However, it will use shared memory to access
data from DolphinSideScript:
    Rewards
    terminals
    states

It will also need to send to DolphinSideScript:
    actions

"""
#Ymem = 108
#Xmem = 200
Ymem = 78
Xmem = 94

class DolphinEnv(gym.Env):
    def __init__(self):

        """
        shared mem is in following format:

        This needs to be changed to this format:

        arr = np.zeros((101,60),dtype=np.float32)

        arr[0][0] = Dtimestep
        arr[0][1] = Etimestep
        arr[0][2] = action
        arr[0][3] = reward
        arr[0][4] = terminal

        arr[1:] = state
        
        """

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(Ymem, Xmem), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(9)
        self.action_space.n = 9

        self.reward_range = (-100, 100)
        self.metadata = None
        self.initialised = False

    def real_init(self):

        with open('pid_num.txt') as f:
            pid = int(f.readlines()[0]) + 1

        self.pid = pid
        self.offset = 0


        #write to file with pid number

        with open('pid_num.txt', 'w') as f:
            f.write(str(self.pid))

        print("My PID: " + str(self.pid))
        
        self.timestep = 0.
        self.init = True
        
        self.data = np.zeros((Ymem + 1,Xmem),dtype=np.float32)
        print("Data Array")
        print(self.data)

        self.shm = shared_memory.SharedMemory(create=True,size=self.data.nbytes,name = 'p' + str(pid))

        print("Saving to shared mem")
        self.shm_array = np.ndarray(self.data.shape, dtype=self.data.dtype, buffer=self.shm.buf)
        self.shm_array[:] = self.data[:]

        print("Launching Dolphin")

        """cmd1 = 'cmd /c C:\\Users\\TYLER\\Downloads\\RLJourney\\DolphinNew\\dolphin'
        cmd2 = '\\Binary\\x64\\Dolphin.exe --no-python-subinterpreters --script C:/Users/TYLER/Downloads/RLJourney/DolphinNew/DolphinSideScript.py \\b --exec="C:\\Users\\TYLER\\Downloads\\GameCollection\\'
        cmd3 = 'SuperSmashBros.Brawl(Europe)(En,Fr,De,Es,It).nkit.gcz"'
        #cmd3 = '\\games\\Mario Kart Wii (USA) (En,Fr,Es).nkit.iso"'
        #cmd /c C:\\Users\\TYLER\\Downloads\\DolphinRevamp\\dolphinScript0\\Dolphin.exe --script C:/Users/TYLER/Downloads/DolphinRevamp/DolphinSideScript.py \\b --exec="C:\\Users\\TYLER\\Downloads\\DolphinRevamp\\dolphinScript0\\games\\NewSuperMarioBros.Wii(Europe)(En,Fr,De,Es,It)(Rev 1).nkit.gcz"

        #launch dolphin
        os.popen(cmd1 + str(pid) + cmd2 + cmd3)"""

        cmd = 'cd ~/Documents/dolphin' + str(pid) + '/build/Binaries && ./dolphin-emu --no-python-subinterpreters\
            --script /home/tyler/Documents/WiiRL/FelkFork/DolphinSideScriptTanks.py\
            --exec="/home/tyler/Documents/GameCollection/Wii Play (Europe) (En,Fr,De,Es,It).nkit.gcz"'

        subprocess.Popen(cmd, shell=True)
        time.sleep(4)

        print("Dolphin Launched Successfully")

    def get_max_episode_steps(self):
        return 1000

    def restart(self):
        with open('pid_num.txt', 'w') as f:
            f.write(str(self.pid))
        
        self.timestep = 0.
        self.init = True

        self.data = np.zeros((Ymem + 1,Xmem),dtype=np.float32)
        print("Data Array")
        print(self.data)

        self.shm = shared_memory.SharedMemory(create=False,size=self.data.nbytes,name = 'p' + str(self.pid))

        print("Saving to shared mem")
        self.shm_array = np.ndarray(self.data.shape, dtype=self.data.dtype, buffer=self.shm.buf)
        self.shm_array[:] = self.data[:]

        print("Launching Dolphin After Crash...")

        cmd1 = 'cmd /c C:\\Users\\TYLER\\Downloads\\RLJourney\\DolphinNew\\dolphin'
        cmd2 = '\\Binary\\x64\\Dolphin.exe --no-python-subinterpreters --script C:/Users/TYLER/Downloads/RLJourney/DolphinNew/DolphinSideScriptTanks.py \\b --exec="C:\\Users\\TYLER\\Downloads\\GameCollection\\'
        cmd3 = 'SuperSmashBros.Brawl(Europe)(En,Fr,De,Es,It).nkit.gcz"'
        #cmd3 = '\\games\\Mario Kart Wii (USA) (En,Fr,Es).nkit.iso"'
        #cmd /c C:\\Users\\TYLER\\Downloads\\DolphinRevamp\\dolphinScript0\\Dolphin.exe --script C:/Users/TYLER/Downloads/DolphinRevamp/DolphinSideScript.py \\b --exec="C:\\Users\\TYLER\\Downloads\\DolphinRevamp\\dolphinScript0\\games\\NewSuperMarioBros.Wii(Europe)(En,Fr,De,Es,It)(Rev 1).nkit.gcz"

        #launch dolphin
        os.popen(cmd1 + str(self.pid) + cmd2 + cmd3)

        time.sleep(4)
                             
    def reset(self):

        if not self.initialised:
            self.real_init()
            self.initialised = True
        #sync
        #print("Resestting...")
        print("Reset Called, PID: " + str(self.pid))
        self.shm_array[0][2 + self.offset] = 0

        if not self.init:
            self.timestep += 1
        else:
            self.init = False
        
        self.shm_array[0][1 + self.offset] = self.timestep

        timer = time.time()
        while True:
            #print(str(self.shm_array[0]) + " " + str(self.shm_array[1])+ " " + str(self.timestep))
            if self.shm_array[0][0 + self.offset] == self.timestep + 1:          
                break

            else:
                if time.time() - timer > 10:
                    print("Waiting 10+ seconds at reset! PID: " + str(self.pid))
                    print("Resestting to t0...")
                    #self.timestep = 0
                    #self.shm_array[0][1 + self.offset] = self.timestep
                    timer = time.time()

        return self.shm_array[1:][:].astype(np.uint8)

    def step(self,action):

        #write timestep and action
        
        self.shm_array[0][2 + self.offset] = action

        self.timestep += 1
        self.shm_array[0][1 + self.offset] = self.timestep

        print("Step Called, PID: " + str(self.pid))
        #wait for new state,reward,terminal
        #sync
        timer = time.time()
        while True:
            time.sleep(0.001)
            #print(str(self.shm_array[0]) + " " + str(self.shm_array[1]) + " " + str(self.timestep))
            if self.shm_array[0][0 + self.offset] == self.timestep + 1:
                break
            else:
                if time.time() - timer > 10:
                    time.sleep(10)
                    print("Dolphin Has likely crashed! No response in 10+ seconds, PID: " + str(self.pid))
                    print(str(self.shm_array[0]) + " " + str(self.shm_array[1]) + " " + str(self.timestep))
                    print("")
                    """os.system("taskkill /f /im Dolphin.exe")
                    time.sleep(5)
                    state = self.shm_array[1:][:].astype(np.uint8)

                    self.restart()

                    return state,0,True,{}"""
                    
        return self.shm_array[1:][:].astype(np.uint8),self.shm_array[0][3 + self.offset],self.shm_array[0][4 + self.offset],{}

def on_press(key):
    global action
    try:
        if key.char == 'q': ####
            action = 0
        elif key.char == 'w':####
            action = 1
        elif key.char == 'e':
            action = 2
        elif key.char == 'r':
            action = 3
        elif key.char == 't':
            action = 4
        elif key.char == 'y':###
            action = 5
        elif key.char == 'u':
            action = 6
        elif key.char == 'i':
            action = 7
        elif key.char == 'o':
            action = 8
        print(action)
    except:pass
        
def on_release(key):
    global action
    action = 9


if __name__ == "__main__":
    with open('pid_num.txt', 'w') as f:
        f.write(str(-1))
    from pynput import keyboard
    start = time.time()
    steps = 1

    envs = []

    for i in range(1):
        env = DolphinEnv()
        #env = gym.wrappers.ResizeObservation(env,(54,100))
        #env = gym.wrappers.FrameStack(env, 3)
        envs.append(env)
        #env = wrap_env(env,4)
        #envs.append(env)


    print(env.observation_space)
    print(env.action_space)
    action = 1
    steps = 1


    """while True:

        for env in envs:
            state = env.reset()
            
        terminal = False
        

        while True:
            for env in envs:
                steps += 1
                state,reward,terminal,_ = env.step(action)
                if terminal:
                    env.reset()
                    print("Fps: " + str(steps / (time.time() - start)))"""

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    tot_reward = 0
    avg_reward = 0
    max_avg_reward = 0
    steps = 1
    while True:

        state = env.reset()
        print("Reset Environment")

        print("Fps: " + str(steps / (time.time() - start)))

        print("Total Reward: " + str(tot_reward))
        print("AVG Reward: " + str(tot_reward / steps))
        print("MAX AVG Reward: " + str(max_avg_reward))

        terminal = False
        trun = False
        tot_reward = 0

        avg_reward = 0
        max_avg_reward = 0

        while not terminal and not trun:
            steps += 1

            state,reward,terminal,_ = env.step(action)
            tot_reward += reward
            if reward != 0:
                print(reward)

            if tot_reward / steps > max_avg_reward:
                max_avg_reward = tot_reward / steps




    
