import win32gui
import win32ui
from ctypes import windll
from PIL import Image
import PIL
from pywinauto import Desktop
import cv2
import numpy as np
import ctypes, time
from copy import copy,deepcopy
import gym
from Region import Region
import pickle
import math
import keyboard
import dxcam
from operator import add

"""
To replace

actions

increase lr slightly??

Template match has HUGE impact on total speed
try to use smaller window if possible
or if not possible get faster template matching

Need to actually implement the region maker

the distance code is redundant, can be removed
look for other stuff to remove

"""

# Bunch of stuff so that the script can send keystrokes to game #

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def KeyPress(key):
    PressKey(keys[key]) # press Q
    time.sleep(.05)
    ReleaseKey(keys[key]) #release Q

def release_keys():
    for key in keys:
        ReleaseKey(keys[key])

def push(key):
    PressKey(keys[key])

def release(key):
    ReleaseKey(keys[key])

keys = {
    "a": 0x1E,
    "b": 0x30,
    "w": 0x11,
    "n": 0x31,
    "m": 0x32,
    "`": 0x29,
    "\\": 0x2B,
    "p": 0x19,
    "e": 0x12,
    "z": 0x2C,
    "c": 0x2E,
    "d": 0x20,
    "f2": 0x3C
    }

#32400 frames/hour!
#ray did 28700
class MarioKartEnv():
    def __init__(self,config=None):

        #self.template = cv2.imread('funky_kong_img2.png')
        #self.template = cv2.cvtColor(self.template, cv2.COLOR_RGB2GRAY)

        self.template = cv2.imread('funky_kong_img3.jpg')
        self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.match_size = 100 #this is in each direction

        #self.orb = cv2.ORB_create() #could look into this
        
        self.tem_w = 36
        self.tem_h = 61

        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(0.0, 1.0, [52, 96])
        self.reward_range = [-1,1]
        self.metadata = None
        
        self.camera = dxcam.create(max_buffer_len=1)
        
        """
        
        1 - accel
        2 - accel+wheely
        3 - accel+drift_hold_right
        4 - accel+drift_hold_left
        5 - accel + right
        6 - accel + left
        7 - accel + drift

        item has been removed

        7 - accel + item
        0 - null
        """
        #yx
        #self.observation_space = gym.spaces.Box(
        #low=0, high=255, shape=(64, 32), dtype=np.uint8)

        save_name = "regions.dat"

        #this should be the same for every map
        self.image_x = 620
        self.image_y = 660
        
        self.grid_size = 10
        self.grid_x = int(self.image_x / self.grid_size)
        self.grid_y = int(self.image_y / self.grid_size)
        
        
        
        self.method = eval('cv2.TM_CCOEFF')

        
        with open(save_name, "rb") as f:
            self.regions = pickle.load(f)

        self.num_chkps = 1 #this is max chkp not num chkp
        for i in self.regions:
            if i.chkp_num > self.num_chkps:
                self.num_chkps = i.chkp_num

        self.total_chkps = self.num_chkps * 3 + 2

        #1720x930
        self.reset()

    def reset(self):
        self.dist = 0
        self.time_till_checkpoint = 1.8
        self.first = True
        release_keys()
        self.held_keys = []
        KeyPress("m") #NEED TO RE-ADD
        #KeyPress("f2")
        time.sleep(0.25)
        self.checkpoint_timer = time.time()
        self.timer = time.time()
        self.prev_action = 0
        self.out_frames = 0
        self.current_chkp = -1
        self.chkp_count = 0

        return self.get_state()[0]

    def template_match(self,img):
        terminal = False
        #crop image so avoid issues -- #og image 2098, 3868
        
        img = img[270:,\
                     1100:]
        #This is 660x620

        
        #print map region code
        """
        cv2.imwrite("map_region_moo.jpg", img)
        raise Exception("stop")
        """
        

        if not self.first:
            
            self.prev_top_left = copy(self.top_left)

            #windowed template matching
            window_top_left = [max(self.prev_center[0] - self.match_size,0),max(self.prev_center[1] - self.match_size,0)]
            match_img = img[window_top_left[1]:min(self.prev_center[1] + self.match_size,620),\
                            window_top_left[0]:min(self.prev_center[0] + self.match_size,660)]

            #cv2.imwrite("match" + str(round(time.time(),4)) + ".jpg", match_img)
            
            res = cv2.matchTemplate(match_img,self.template,self.method)#img
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            #elementwise addition of two lists
            self.top_left = list(map(add, window_top_left,list(max_loc)))

            y_dif = list(self.prev_top_left)[1] - list(self.top_left)[1]
            x_dif = list(self.top_left)[0] - list(self.prev_top_left)[0]
        else:

            #full template matching for first time
            res = cv2.matchTemplate(img,self.template,self.method)#img
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            self.top_left = max_loc
            self.prev_top_left = copy(self.top_left)
            self.prev_center = [self.prev_top_left[0] + int(self.tem_w / 2),\
                                self.prev_top_left[1] + int(self.tem_h / 2)]
            
            self.first = False
            return 0,False

    
        self.dist = x_dif**2 + y_dif**2

        #exception for broken template matching
        if self.dist > 950:#

            #need to allow it to refind template next frame
            self.first = True
            reward = 0
            bottom_right2 = (self.prev_top_left[0] + self.tem_w, self.prev_top_left[1] + self.tem_h)
            cv2.rectangle(img,self.prev_top_left, bottom_right2, 255, 2)
            
            bottom_right = (self.top_left[0] + self.tem_w, self.top_left[1] + self.tem_h)
            cv2.rectangle(img,self.top_left, bottom_right, 255, 2)
            
            #cv2.imwrite("match_error" + str(round(time.time(),1)) + ".jpg", img)

            self.dist = 0
        else:

            #region code - #cropped image 1220, 880 (y,x)
            reward = self.get_reward(x_dif,y_dif)
            if self.out_frames > 3:
                terminal = True
                reward = -1
            
            """bottom_right = (self.top_left[0] + self.tem_w, self.top_left[1] + self.tem_h)
            cv2.rectangle(img,self.top_left, bottom_right, 128, 2)
            cv2.imwrite("bug_test" + str(time.time()) + ".jpg", img)

            raise Exception("stop")"""

        return reward,terminal

    def get_reward(self,x_dif,y_dif):
        reward = 0
        reset_frames = True

        #this are based off funky's face
        add_x = int(self.tem_w / 2)
        add_y = int(self.tem_h / 2)

        #center location
        x = self.top_left[0] + add_x
        y = self.top_left[1] + add_y
        self.prev_center = [x,y]

        x = math.floor(x / self.grid_size)
        y = math.floor(y / self.grid_size)

        #get which grid cell
        num = self.convert_xy_to_num(x,y)

        #check out of bounds
        if not self.regions[num].in_bounds:
            self.out_frames += 1
            reset_frames = False

        #check dir_x
        #reward += x_dif * self.regions[num].dir_x

        #check dir_y
        #reward -= y_dif * self.regions[num].dir_y

        #checkpoints
        if self.regions[num].is_chkp:
            if self.regions[num].chkp_num > self.current_chkp or (self.regions[num].chkp_num == 0 and self.current_chkp == self.num_chkps):
                reward += 1 #replace this
                self.chkp_count += 1
                self.checkpoint_timer = time.time()
                self.current_chkp = self.regions[num].chkp_num
                if self.current_chkp == self.num_chkps:
                    #lap complete
                    self.time_till_checkpoint -= 0.25
                    
            """
            elif self.regions[num].chkp_num < self.current_chkp or \
                 (self.regions[num].chkp_num == self.num_chkps and (self.current_chkp == 0 or self.current_chkp == -1)):
                
                self.out_frames += 1
                reset_frames = False
            """
        
        if reset_frames:
            self.out_frames = 0

        #timer for reaching checkpoints
        if time.time() - self.checkpoint_timer > self.time_till_checkpoint:
            self.out_frames = 10

        if self.chkp_count > self.total_chkps:
            self.out_frames = 10
            
        return reward

    def convert_xy_to_num(self,x,y):
        return x + y * self.grid_x

    def is_inside(self,point,reg_point,reg_end_point):
        #loop over xy
        for i in range(2):
            if not (point[i] >= reg_point[i] and point[i] <= reg_end_point[i]):
                return False
        return True        

    def get_state(self):

        #self.camera.start(region=(100, 70, 1820, 1000),target_fps=60)
        
        im = self.camera.grab()
        while im is None:
            im = self.camera.grab()
        im = im[70:1000,100:1820]
        #cv2.imwrite("test0" + str(time.time()) + ".jpg", im)
        #raise Exception("akshf")
        
        #gets the top_left var
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


        #check gone off edge - this should be replaced tho with lower point
        if im[178,1483] == 0:
            terminal = True
            reward = -1
        else:
            terminal = False


        if not terminal:
            reward,terminal = self.template_match(im)


        #im = im[80:, 85: 3868 - 85]
        
        im = cv2.resize(im, (96,52), interpolation = cv2.INTER_AREA)
        #cv2.imwrite("test1" + str(time.time()) + ".jpg", im)
        im = self.process_frame(im)


        return im,reward,terminal

    def step(self,action=0):
        #time.sleep(0.005)
        terminal = False

        self.apply_action(action)
        #press some key
        state,reward,terminal = self.get_state()

        info = {}

        return state,reward,terminal,info

    def process_frame(self,frame):
        #could try using half precision if needed
        frame = np.true_divide(frame, 255).astype(np.float32)
        return frame        

    def apply_action(self,action):
        """
        
        1 - accel
        2 - accel+wheely
        3 - accel+drift_hold_right
        4 - accel+drift_hold_left

        0 - null
        5 - accel + right
        6 - accel + left
        7 - accel + item
        """
        self.prev_held = copy(self.held_keys)

        #null action removed
        action += 1
        
        """if action == 0:
            self.held_keys = []"""
        if action == 1:#0
            self.held_keys = ["w"]
        elif action == 2:#1
            self.held_keys = ["w","e"]
        elif action == 3:#2
            self.held_keys = ["w","c","d"]
        elif action == 4:#3
            self.held_keys = ["w","c","a"]
        elif action == 5:#4
            self.held_keys = ["w","d"]
        elif action == 6:#5
            self.held_keys = ["w","a"]
        elif action == 7:#6
            self.held_keys = ["w","c"]
        """elif action == 7:
            self.held_keys = ["w","z"]"""

        for i in self.held_keys:
            if i not in self.prev_held:
                push(i)

        for i in self.prev_held:
            if i not in self.held_keys:
                release(i)

        #print()

if __name__ == "__main__":
    time.sleep(5)
    env = MarioKartEnv()
    state = env.reset()
    score = 0
    action = 0
    steps = 0
    starter = time.time()
    while True:
        steps += 1

        if keyboard.is_pressed('i'):
            action = 1
        elif keyboard.is_pressed('k') and keyboard.is_pressed('n'):
            action = 2
        elif keyboard.is_pressed('h') and keyboard.is_pressed('n'):
            action = 3
        elif keyboard.is_pressed('k'):
            action = 4
        elif keyboard.is_pressed('h'):
            action = 5
        elif keyboard.is_pressed('n'):
            action = 6
        else:
            action = 0

        
        state,reward,terminal,info = env.step(action)

        if steps % 500 == 0:
            print("Frames per Hour:")
            time_seg = 500 / (time.time() - starter)
            
            print(time_seg * 3600)
            starter = time.time()
        
        score += reward
        #print(reward)
        if terminal:
            print("Total Reward: " + str(score))
            score = 0
            env.reset()
