from multiprocessing.connection import Listener,Client
from multiprocessing import get_context
import numpy as np
import time
import os
import subprocess
import gym
import cv2
import warnings
import signal
import random
from Wrappers import wrap_env
from PIL import Image

warnings.filterwarnings("ignore")

"""
This program implements the standard gym MDP

However, it will use multiprocessing connection to access
data from DolphinSideScript:
    Rewards
    terminals
    states

It will also need to send to DolphinSideScript:
    actions

"""


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

class DolphinEnvVec(gym.Env):
    def __init__(self,num_envs=2):

        #Env just sends single int - action

        #self.data is a list, not an ndarray

        #DolphinSideScript sends back reward,done,state
        #self.data[0][0] = reward - int
        #self.data[0][1] = reward - dec
        #self.data[0][2] = done
        #self.data[1] = state

        self.state_size_x = 140#168#112
        self.state_size_y = 75#90#60
        print("Initialised a Dolphin Environment")
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, self.state_size_y, self.state_size_x), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)

        #self.reward_range = (-100, 100)
        #self.metadata = None
        self.initialised = False

        self.num_envs = num_envs

        self.script_pids = [None for i in range(num_envs)]

        self.l_conns = [None for i in range(num_envs)]
        self.c_conns = [None for i in range(num_envs)]


    def get_action_meanings(self):
        return ['NOOP','LEFT','RIGHT','UP','DOWN','AIMRIGHT','AIMLEFT','AIMUP','AIMDOWN','FIRE']

    def real_init(self,pid,restart=False):

        with open('pid_num.txt', 'w') as f:
            f.write(str(pid))

        self.action = 0

        print("Instance PID: " + str(pid))

        self.init = True

        print("Launching Dolphin")
        time.sleep(0.25)
        self.launch(pid)
        print("Dolphin Launched Successfully")

        print("Creating Listener...")
        addressListener = ('localhost', 26330 + pid)
        listener = Listener(addressListener, authkey=b'secret password')
        print("Waiting for scripside connection...")


        self.l_conns[pid] = listener.accept()
        msg = self.l_conns[pid].recv()
        print("Listener Connection accepted, message: " + str(msg))

        addressClient = ('localhost', 25330 + pid)
        self.c_conns[pid] = Client(addressClient, authkey=b'secret password')
        time.sleep(.5)
        self.c_conns[pid].send("Start, from Main Env")
        print("Created Client")

        if restart:
            print("Waiting for messages in restart...")

            msg = self.l_conns[pid].recv()
            if msg != "RESET":
                print("Got not Reset!")
                print(msg)

            self.c_conns[pid].send("RESET")

        print("Initialisation Complete")

    def launch(self,pid):

        gamename = "Items"
        gamefile = 'Mario Kart Wii (USA) (En,Fr,Es).nkit.iso"'

        cmd1 = 'cmd /c C:\\Users\\Tyler\\Documents\\RL2\\effRainbow' + gamename + '\\dolphin'
        cmd2 = '\\Binary\\x64\\Dolphin' + str(pid) + '.exe --no-python-subinterpreters --script C:/Users/Tyler/Documents/RL2/effRainbow' \
               + gamename + '/DolphinSideScript' + gamename + '.py \\b\
         --exec="C:\\Users\\Tyler\\Documents\\RL2\\GameCollection\\'

        # launch dolphin
        os.popen(cmd1 + str(pid) + cmd2 + gamefile)

        print("Launched Envs")
        time.sleep(1.5)

        with open('C:/Users/Tyler/Documents/RL2/effRainbow' + gamename + '/script_pid' + str(pid) + '.txt') as f:
            self.script_pids[pid] = int(f.readlines()[0])
            print(self.script_pids[pid])

        time.sleep(1)

    def get_max_episode_steps(self):
        return 100000

    def restart(self,i):
        self.real_init(i,restart = True)

    def reset(self):

        if not self.initialised:
            for i in range(self.num_envs):
                self.real_init(i)
            self.initialised = True

        if self.init:
            self.init = False

        self.action = 0

        print("Waiting for messages in reset...")

        for i in range(self.num_envs):
            msg = self.l_conns[i].recv()
            if msg != "RESET":
                print("Got not Reset!")
                print(msg)

        for i in range(self.num_envs):
            self.c_conns[i].send("RESET")

        self.states = []
        for i in range(self.num_envs):
            timer = time.time()
            resent = False
            running = True
            while running:
                while self.l_conns[i].poll():
                    try:
                        msg = self.l_conns[i].recv()
                        running = False
                    except:
                        pass

                if time.time() - timer > 3 and not resent:
                    print("Waiting in Reset!")
                    resent = True
                    self.c_conns[i].send(self.action)

                elif time.time() - timer > 10:
                    print("Crash at Reset, restarting...")
                    subprocess.check_output("Taskkill /PID %d /F" % self.script_pids[i])
                    time.sleep(5)
                    self.restart(i)
                    time.sleep(0.5)

            print("Got States!")
            self.state = LazyFrames(msg[1]) # This is stored for crashes
            self.states.append(LazyFrames(msg[1]))

        print("Finished Collecting States!")
        return self.states

    def step(self,actions):


        #write action
        for i in range(self.num_envs):
            self.c_conns[i].send(actions[i])

        #print("Waiting for message in step...")
        self.states = []
        self.rewards = []
        self.dones = []
        self.truns = []

        for i in range(self.num_envs):
            timer = time.time()
            running = True
            resent = False
            failed = False
            while running:
                while self.l_conns[i].poll():
                    try:
                        msg = self.l_conns[i].recv()

                        running = False
                        break
                    except Exception as e:
                        print(e)
                        print("Crash at step, Connection failed, restarting...")
                        try:
                            subprocess.check_output("Taskkill /PID %d /F" % self.script_pids[i])
                        except:pass
                        time.sleep(3)
                        self.restart(i)
                        self.states.append(self.state)  # This is not a real state, just a dummy
                        self.rewards.append(0)
                        self.dones.append(False)
                        self.truns.append(True)
                        running = False
                        failed = True
                        time.sleep(1)
                        break

                if time.time() - timer > 3 and not resent:
                    print("Waiting in Step!")
                    resent = True
                    self.c_conns[i].send(actions[i])

                elif time.time() - timer > 10:
                    #restart
                    print("Crash at step, restarting...")
                    subprocess.check_output("Taskkill /PID %d /F" % self.script_pids[i])
                    #os.kill(self.script_pids[i], signal.SIGTERM)
                    time.sleep(4)
                    self.restart(i)
                    self.states.append(self.state) #This is not a real state, just a dummy
                    self.rewards.append(0)
                    self.dones.append(False)
                    self.truns.append(True)
                    failed = True
                    time.sleep(1)
                    break

            if not failed:
                if msg[0][2] or msg[0][3]:
                    reset_check = self.l_conns[i].recv()
                    if reset_check != "RESET":
                        print("Got not Reset!")

                    self.c_conns[i].send("RESET")

                if msg[1] is not None:
                    self.states.append(LazyFrames(msg[1]))
                else:
                    self.states.append(msg[1])
                reward = self.convert_reward(int(msg[0][0]),int(msg[0][1]))
                self.rewards.append(reward)
                self.dones.append(msg[0][2])
                self.truns.append(msg[0][3])

        return self.states,self.rewards,self.dones,self.truns,{}

    def convert_reward(self,inte,deci):
        reward = round(float(inte + deci / 100000),5)
        return reward

def on_press(key):
    global action
    try:
        if key.char == '1':
            action = 1
        elif key.char == '2':
            action = 2
        elif key.char == '3':
            action = 3
        elif key.char == '4':
            action = 4
        elif key.char == '5':
            action = 5
        elif key.char == '6':
            action = 6
        elif key.char == '7':
            action = 7
        elif key.char == '8':
            action = 8
        elif key.char == '9':
            action = 9

    except:pass
        
def on_release(key):
    global action
    action = 0

if __name__ == "__main__":
    with open('pid_num.txt', 'w') as f:
        f.write(str(0))
    from pynput import keyboard
    start = time.time()
    steps = 1

    envs = []

    """for i in range(2):
        env = DolphinEnv()
        #env = gym.wrappers.ResizeObservation(env,(54,100))
        #env = gym.wrappers.FrameStack(env, 3)
        envs.append(env)
        #env = wrap_env(env,4)
        #envs.append(env)"""

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    vec_envs = DolphinEnvVec(1)
    obs = vec_envs.reset()
    states = None
    action = 0
    total_reward = 0
    while True:
        time.sleep(0.03)
        states_,rewards,terminals,trun,_ = vec_envs.step([action])
        total_reward += rewards[0]
        if abs(rewards[0]) > 0.1:
            print("Reward: " + str(rewards[0]) + ", Total Reward: " + str(total_reward))
        if terminals[0]:
            print("Episode Reward Reward: " + str(total_reward))
            print("\n\n\n\n\n")
            total_reward = 0
        """print(states_)
        print(len(states_))
        print(states_[0].shape)

        if terminals[0] == 1:
            cv2.imwrite("state.jpg", states[0])
            cv2.imwrite("state_.jpg", states_[0])
            raise Exception("Stop")"""

        states = states_

    state = i.reset()
    ####Loop Version
    while True:
        for i in envs:
            state,reward,terminal,_ = i.step(random.randint(1,9))

            if terminal:
                state = i.reset()

    print(env.observation_space)
    print(env.action_space)
    action = 1
    real_action = 1
    action_count = 0
    steps = 1

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


            if action_count > 4:
                real_action = action
                action_count = 0
            state,reward,terminal,_ = env.step(action)#real_action
            action_count += 1
            tot_reward += reward
            #if reward != 0:
            print(reward / 4)

            if tot_reward / steps > max_avg_reward:
                max_avg_reward = tot_reward / steps




    
