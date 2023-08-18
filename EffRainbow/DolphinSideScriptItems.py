
#Window is 500x270 when captured
with open('logg.txt', 'w') as f:
    f.write("Created Fresh Logg")

import sys
sys.path.append("C:\\Users\\TYLER\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages")
#sys.path.append("C:\\ProgramData\\Anaconda3\\Lib\\site-packages")

import os
with open('logg.txt', 'a') as f:
    f.write("\nImported Os\n")
    
with open('pid_num.txt') as f:
    pid = int(f.readlines()[0])

with open('script_pid' + str(pid) + '.txt', 'w') as f:
    f.write(str(os.getpid()))

with open('logg.txt', 'a') as f:
    f.write("\nPID:" + str(pid))
    f.write("Real PID: " + str(os.getpid()))

from multiprocessing.connection import Listener,Client
import numpy as np
from copy import deepcopy
with open('logg.txt', 'a') as f:
    f.write('half libraries installed\n')

from PIL import Image,ImageEnhance
import math
import time
from collections import deque
from statistics import mean
import random
from threading import Thread

with open('logg.txt', 'a') as f:
    f.write('Got Pid ' + str(pid) + '\n')

#Ymem = 78
#Xmem = 94

Ymem = 114
Xmem = 208

from dolphin import event, gui,savestate,memory,controller
with open('logg.txt', 'a') as f:
    f.write('\nImported FelkLibs')

class DolphinSideEnv():
    def __init__(self):
        ########### Game Code

        self.last_action = 0

        self.k = 4 #This is the number of stacked frames
        self.frames = deque([], maxlen=self.k) #frame buffer for framestacking
        
        ##################### End Game Code
        
        self.window_header = 40
        self.window_width = 100
        self.window_height = 60

        self.frameskip = 4
        self.timestep = 0

        self.current_stage = 4
        self.ep_length = 0
        self.last_action = 0

        with open('logg.txt', 'a') as f:
            f.write('\nAbout to make Client\n')

        addressClient = ('localhost', 26330 + pid)
        self.c_conn = Client(addressClient, authkey=b'secret password')
        self.c_conn.send("Start, from Scripter")

        with open('logg.txt', 'a') as f:
            f.write('About to make Listener\n')

        addressListener = ('localhost', 25330 + pid)
        listener = Listener(addressListener, authkey=b'secret password')
        self.l_conn = listener.accept()
        msg = self.l_conn.recv()

        with open('logg.txt', 'a') as f:
            f.write('Accepted Connection from main env, message: ' + str(msg) + '\n')

        #create data array

        self.reset()

        with open('logg.txt', 'a') as f:
            f.write('Init Reset Successful\n')

        self.c_conn.send("RESET")

        msg = self.l_conn.recv()
        if msg != "RESET":
            with open('logg.txt', 'a') as f:
                f.write("NOT RESET!")

    def reset(self):

        time.sleep(0.01)

        self.ep_length = 0
        self.action = 0
        self.next_race_checkpoint = 1.05
        self.frames_since_checkpoint = 0
        self.is_terminal = False
        self.last_action = 0
        self.alternator = False
        self.alternator2 = False
        self.prev_speeds = deque([100,100,100,100,100,100,100,100,100,100], 40)

        savestate.load_from_slot(random.randint(1,4))

        self.cur_dir = 0
        self.dir_frames = 0
        
        ##################### End Game Code


        #check reset

        #write state
        self.data = [[None,None,None,None],None]

        self.dic = {
            "Left": False,
            "Right": False,
            "Down": False,
            "Up": False,
            "Z": False,
            "R": False,
            "L": False,
            "A": True,
            "B": False,
            "X": False,
            "Y": False,
            "Start": False,
            "StickX": 128,
            "StickY": 128,
            "CStickX": 128,
            "CStickY": 128,
            "TriggerLeft": 0,
            "TriggerRight": 0,
            "AnalogA": 0,
            "AnalogB": 0,
            "Connected": True
        }

    def get_reward_terminal(self):
        # Returns reward,terminal,trun

        controller.set_gc_buttons(0, self.dic)

        ########### Game Code
        terminal = False
        reward = 0

        speed = memory.read_f32(0x8111ECD4)
        race_completion = memory.read_f32(0x81115258)

        if self.ep_length % 15 == 0:
            self.prev_speeds.append(speed)
        self.frames_since_checkpoint += 1

        if race_completion > self.next_race_checkpoint:
            self.next_race_checkpoint += 0.05

            reward += 0.5 - min(0.25, self.frames_since_checkpoint * 0.0025)
            self.frames_since_checkpoint = 0

        if race_completion > 4:
            terminal = True
            reward += 9.75

        if speed > 100:
            reward += 0.0075
        elif speed > 95:
            reward += 0.004
        elif speed > 92:
            reward += 0.003
        elif speed > 90:
            reward += 0.0025
        elif speed > 87:
            reward += 0.002
        elif speed > 85:
            reward += 0.001
        elif speed > 82:
            reward += 0.00075
        elif speed > 78:
            reward += 0.0005
        elif speed > 70:
            reward += 0.0001
        elif speed < 65:
            reward -= 0.0001
        elif speed < 60:
            reward -= 0.001
        elif speed < 50:
            reward -= 0.005
        elif speed < 35:
            reward -= 0.01

        if max(list(self.prev_speeds)) < 45 and not self.is_terminal:
            terminal = True
            reward = -0.1
            self.is_terminal = True

        ##################### End Game Code

        return reward,terminal,False

    def apply_action(self,action):

        self.last_action = action

        self.dic = {"Left": False,"Right": False,"Down": True,
            "Up": False,"Z": False,"R": False,"L": False,
            "A": True,"B": False,"X": False,"Y": False,
            "Start": False,"StickX": 128,"StickY": 128,"CStickX": 128,
            "CStickY": 128,"TriggerLeft": 0,"TriggerRight": 0,
            "AnalogA": 0,"AnalogB": 0,"Connected": True
        }

        #if the sub-if conditions are met, it will just go striaght for a frame

        self.held_reward = 0
        if action == 0: #forward-wheelie
            self.cur_dir = 0
            self.dic["Down"] = False
            if self.alternator:
                self.dic["Up"] = True
            self.alternator = not self.alternator

        elif action == 1: #hLeft
            if self.cur_dir == -1 or self.cur_dir == 0:
                self.dic["R"] = True
                self.dic["StickX"] = 0
                self.cur_dir = -1
            else:
                self.cur_dir = 0

        elif action == 2: #sLeft
            if self.cur_dir == -1:
                self.dic["R"] = True
                self.dic["StickX"] = 255
            elif self.cur_dir == 0:
                self.dic["R"] = True
                self.dic["StickX"] = 0
                self.cur_dir = -1
            elif self.cur_dir == 1:
                self.cur_dir = 0

        elif action == 4: #hRight
            if self.cur_dir == 1 or self.cur_dir == 0:
                self.dic["R"] = True
                self.dic["StickX"] = 255
                self.cur_dir = 1
            else:
                self.cur_dir = 0

        elif action == 3: #sRight
            if self.cur_dir == 1:
                self.dic["R"] = True
                self.dic["StickX"] = 0
            elif self.cur_dir == 0:
                self.dic["R"] = True
                self.dic["StickX"] = 255
                self.cur_dir = 1
            elif self.cur_dir == -1:
                self.cur_dir = 0

        elif action == 5: # item
            self.dic["A"] = False
            if self.alternator2:
                self.dic["L"] = True
            else:
                self.dic["L"] = True
            self.alternator2 = not self.alternator2
        """
        elif action == 3:
            self.dic["R"] = True
            self.dic["StickX"] = 96
        elif action == 4:
            self.dic["R"] = True
            self.dic["StickX"] = 164
        """

        controller.set_gc_buttons(0, self.dic)
        
    def step(self):
        self.ep_length += 1

        timer = time.time()
        running = True
        while running:
            while self.l_conn.poll():
                action = self.l_conn.recv()
                running = False

            if time.time() - timer > 3:
                with open('logg.txt', 'a') as f:
                    f.write("Step Function waiting too long!")
                action = 0
                break

        self.apply_action(action)

    def step2(self,reward,terminal,trun,image):

        # internetal res is 640x528
        # at internet res, frame dump gives 832x456

        #transmit quite large - allow scaling on other side
        #suggestion - 208x114? ( div by 8)

        """if terminal:
            self.data[1] = None
            self.data[0][2] = int(False)
            for i in range(10):
                self.c_conn.send(self.data)"""

        image = self.process_frame(image)

        #prepare data into array data
        self.data[0][2] = int(terminal)
        self.data[0][3] = int(trun)
        reward_int,reward_dec = self.convert_reward(reward)
        self.data[0][0] = reward_int
        self.data[0][1] = reward_dec
        self.data[1] = image

        # reset if needed
        if terminal or trun:
            copied_data = deepcopy(self.data)
            self.reset()
            return copied_data
        else:
            #send data
            self.c_conn.send(self.data)

    def sync_reset(self,obs,img):
        obs[1] = self.process_frame(img,terminal=True)
        self.c_conn.send(obs)

        self.c_conn.send("RESET")

        msg = self.l_conn.recv()
        if msg != "RESET":
            with open('logg.txt', 'a') as f:
                f.write("NOT RESET!")

    def process_frame(self,image,terminal=False):
        # internetal res is 640x528
        # at internet res, frame dump gives 832x456

        image = image.convert("L")
        #image = image.crop((64,16,768,392))
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(4)
        #140x75
        image = image.resize((140, 75)) #(112,60),(168,90)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(4)

        image = np.asarray(image)

        #process Uint
        image = image.astype(np.uint8)
        image = np.expand_dims(image, axis=2)

        #Convert to PyTorch Image
        image = np.swapaxes(image,2,0)

        #Framestack
        if terminal or len(self.frames) != 4:
            for i in range(self.k):
                self.frames.append(image) #self.frames has max length 4, so this just resets it
        else:
            self.frames.append(image)

        observation = list(self.frames)

        return observation

    def convert_reward(self,reward):
        reward_int = math.floor(reward)
        reward_dec = round(reward % 1, 4)
        reward_dec = int(100000 * reward_dec)
        return reward_int, reward_dec

next_stage = False

with open('script_pid' + str(pid) + '.txt', 'w') as f:
    f.write(str(os.getpid()))

time.sleep(0.1)

for i in range(4):
    await event.frameadvance()

env = DolphinSideEnv()

for i in range(env.frameskip):
    await event.frameadvance()

reward = 0
terminal = False
trun = False
red = 0xffff0000

with open('logg.txt', 'a') as f:
    f.write('\nEntering Main While loop')

while True:
    env.step()

    for i in range(env.frameskip):
        (width,height,data) = await event.framedrawn()

        rewardN,terminalN,trunN = env.get_reward_terminal()

        if not terminal:
            terminal = terminal or terminalN
            trun = trun or trunN
            reward += rewardN

        if terminal or trun or next_stage:
            for i in range(2):
                await event.frameadvance()

    img = Image.frombytes('RGBA', (width,height), data, 'raw')

    obs = env.step2(reward,terminal,trun,img) #resetting can happen here

    if terminal or trun:
        (width, height, data) = await event.framedrawn()
        img = Image.frombytes('RGBA', (width, height), data, 'raw')

        env.sync_reset(obs,img)

    #gui.draw_text((10, 10), red, f"HI")
    
    reward = 0
    terminal = False
    trun = False
        

