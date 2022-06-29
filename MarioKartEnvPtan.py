#import win32gui
#import win32ui
from ctypes import windll
from PIL import Image
import PIL
#from pywinauto import Desktop
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
    "l": 0x26,
    "k": 0x25,
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

        self.action_space = gym.spaces.Discrete(5)
        #self.observation_space = gym.spaces.Box(0.0, 1.0, [32, 64])

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(930, 1720, 3), dtype=np.float32)
        
        self.reward_range = [-1,1]
        self.metadata = None
        
        self.camera = dxcam.create(max_buffer_len=1)

        self.fps = 30
        self.time_per_step = 1/self.fps

        
        
        """
        
        1 - accel
        2 - accel+wheely
        3 - hard right
        4 - hard left
        5 - soft right
        6 - soft left

        """
        #yx

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

        self.focal_points = self.get_focal_points(self.num_chkps + 1)

        #1720x930
        self.time_till_checkpoint = 1.2
        self.reset()

    def get_focal_points(self,num_focals):
        temp_list = []
        for i in range(num_focals):
            for j in self.regions:
                if j.chkp_num == i and j.focal:
                    temp_list.append([j.x,j.y])

        #add extra at start for when we are at last checkpoint
        for j in self.regions:
            if j.chkp_num == 0 and j.focal:
                temp_list.append([j.x,j.y])        
                    
        return temp_list
        

    def reset(self):
        
        self.fps_timer = time.time()
        self.dist = 0
        self.drift_held = 0
        #self.time_till_checkpoint = 0.4
        self.first = True
        self.wheel_alt = False
        release_keys()
        self.direction = 0 
        self.held_keys = []
        KeyPress("m") #NEED TO RE-ADD
        KeyPress("f2")
        time.sleep(0.25)
        self.checkpoint_timer = time.time()
        self.timer = time.time()
        self.prev_action = 0
        self.out_frames = 0
        self.current_chkp = -1
        self.chkp_count = 0
        self.c_frames = 0
        self.avg_time = np.array([0.3,0.3,0.3],dtype=np.float32)
        self.avg_dist = np.array([10,10,10,10,10],dtype=np.float32)
        self.test_timer = time.time()
        self.rewards_claimed = [False,False,False,False,False,False,False,False]
        self.held_reward = 0

        return self.get_state()[0]

    def template_match(self,img):
        terminal = False
        #crop image so avoid issues -- #og image 2098, 3868
        
        img = img[270:,\
                     1100:]
        #This is 660x620

        
        #print map region code
        
        #cv2.imwrite("map_region_yoshi_falls.jpg", img)
        #raise Exception("stop")
        
        
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

            old_dist_to_focal = np.sqrt((list(self.prev_top_left)[1] - self.focal_points[self.current_chkp + 1][1])**2 +\
                                (list(self.prev_top_left)[0] - self.focal_points[self.current_chkp + 1][0])**2)

            new_dist_to_focal = np.sqrt((list(self.top_left)[1] - self.focal_points[self.current_chkp + 1][1])**2 +\
                                (list(self.top_left)[0] - self.focal_points[self.current_chkp + 1][0])**2)

            reward = (old_dist_to_focal - new_dist_to_focal) / 12
            reward = reward ** 2
            #change = np.sqrt(x_dif**2 + y_dif**2) / 100
            #reward = (reward + change)**2
            
            #print(change)

            #reward = reward ** 2 if reward > 0 else -(reward ** 2)
            #reward = reward * 3
            
        else:
            #cv2.imwrite("first" + str(round(time.time(),1)) + ".jpg", img)
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
        
        self.avg_dist = np.roll(self.avg_dist,1)
        self.avg_dist[0] = self.dist
        

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
            reward += self.get_reward(x_dif,y_dif)
            if self.out_frames > 3:
                terminal = True
                if reward < 2:
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
        """if not self.regions[num].in_bounds:
            self.out_frames += 0
            reset_frames = False"""

        #check dir_x
        #reward += x_dif * self.regions[num].dir_x

        #check dir_y
        #reward -= y_dif * self.regions[num].dir_y

        #checkpoints
        if self.regions[num].is_chkp:
            if not(self.regions[num].chkp_num > 5 and (self.current_chkp == -1 or self.current_chkp == 0)):
                if self.regions[num].chkp_num > self.current_chkp or \
                   (self.regions[num].chkp_num == 0 and self.current_chkp == self.num_chkps):
                    reward += .5
                    self.chkp_count += 1
                    #print(time.time() - self.checkpoint_timer)
                    self.avg_time = np.roll(self.avg_time,1)
                    self.avg_time[0] = time.time() - self.checkpoint_timer
                    self.checkpoint_timer = time.time()
                    self.current_chkp = self.regions[num].chkp_num
                    if self.current_chkp == self.num_chkps:
                        #lap complete
                        #print(self.c_frames)
                        reward = 5
                    
            """
            elif self.regions[num].chkp_num < self.current_chkp or \
                 (self.regions[num].chkp_num == self.num_chkps and (self.current_chkp == 0 or self.current_chkp == -1)):
                
                self.out_frames += 1
                reset_frames = False
            """

        #just to prevent going backwards
        if self.chkp_count > 0:
            #print((np.mean(self.avg_time) + (time.time() - self.checkpoint_timer)) / 4)
            #timer for reaching checkpoints
            if (np.mean(self.avg_time) + (time.time() - self.checkpoint_timer)) / 4 > self.time_till_checkpoint:
                self.out_frames = 10     

        if self.chkp_count > self.total_chkps:
            reward = 15
            self.out_frames = 10

        #print(np.mean(self.avg_dist))
        if np.mean(self.avg_dist) < 1.0 and self.c_frames > 30:
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
        im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


        #check gone off edge - this should be replaced tho with lower point
        if im_grey[178,1483] == 0:
            terminal = True
            reward = -1
        else:
            terminal = False


        if not terminal:
            reward,terminal = self.template_match(im_grey)

        #im = im[80:, 85: 3868 - 85]
        
        #im = cv2.resize(im, (94,56), interpolation = cv2.INTER_AREA)
        im = np.expand_dims(im, axis=0)
        #cv2.imwrite("test1" + str(time.time()) + ".jpg", im)

        #ptan divides for you
        #im = self.process_frame(im)


        return im,reward,terminal

    def reset_claimed_rewards(self):
        for i in range(len(self.rewards_claimed)):
            self.rewards_claimed[i] = False

    def step(self,action=0):
        #time.sleep(0.005)
        terminal = False

        self.apply_action(action)
        #press some key
        state,reward,terminal = self.get_state()

        if self.drift_held > 21 and not self.rewards_claimed[0]:
            reward += 0.15
            self.rewards_claimed[0] = True

        elif self.drift_held > 25 and not self.rewards_claimed[1]:
            reward += 0.2
            self.rewards_claimed[1] = True

        elif self.drift_held > 30 and not self.rewards_claimed[2]:
            reward += 0.4
            self.rewards_claimed[2] = True

        elif self.drift_held > 35 and not self.rewards_claimed[3]:
            reward += 0.6
            self.rewards_claimed[3] = True
            
        elif self.drift_held > 40 and not self.rewards_claimed[4]:
            reward += 1.5
            self.rewards_claimed[4] = True

        elif self.drift_held > 50 and not self.rewards_claimed[5]:
            reward -= .2
            self.rewards_claimed[5] = True

        elif self.drift_held > 55 and not self.rewards_claimed[6]:
            reward -= .4
            self.rewards_claimed[6] = True

        elif self.drift_held > 60 and not self.rewards_claimed[7]:
            reward -= .6
            self.rewards_claimed[7] = True

        elif self.drift_held > 65:
            reward -= .02

        self.c_frames += 1
        #regulate fps
        reward += self.held_reward
        self.held_reward = 0
        
        while time.time() < self.fps_timer + self.time_per_step:
            pass
        self.fps_timer = time.time()

        info = {}

        #if terminal:
        #print("Recorded FPS: " + str(self.c_frames / (time.time() - self.test_timer)))

        return state,reward,terminal,info

    def process_frame(self,frame):
        #could try using half precision if needed
        frame = np.true_divide(frame, 255).astype(np.float32)
        return frame

    def end_drift(self,bonus=False):
        self.drift_held = 0
        if self.drift_held > 40:
            self.held_reward = 1.2
            if bonus:
                self.held_reward = 2
        if self.drift_held > 0:
            self.reset_claimed_rewards()

                
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
        #action += 2
        """
        if action == 1:#0 forward
            self.held_keys = ["w"]
            self.direction = 0
        """
            
        if action == 0:#1
            self.direction = 0
            self.end_drift(bonus=True)
                
            if self.wheel_alt:
                self.held_keys = ["w","e"]
            else:
                self.held_keys = ["w"]
            self.wheel_alt = not self.wheel_alt

        #hard drifts
        elif action == 1:#
            if self.direction == 1 or self.direction == 0:
                self.held_keys = ["w","c","d"]
                self.direction = 1
                self.drift_held += 1.05
            else:
                self.held_keys = ["w"]
                self.direction = 0
                self.end_drift()
                
        elif action == 2:#3
            if self.direction == -1 or self.direction == 0:
                self.held_keys = ["w","c","a"]
                self.direction = -1
                self.drift_held += 1.05
            else:
                self.held_keys = ["w"]
                self.direction = 0
                self.end_drift()

        #soft drifts
        elif action == 3:#right turn
            if self.direction == 1:
                self.held_keys = ["w","c","a"]
                self.drift_held += .5
            elif self.direction == 0:
                self.held_keys = ["w","c","d"]
                self.drift_held += 1.05
                self.direction = 1
            else:
                self.held_keys = ["w"]
                self.direction = 0
                self.end_drift()
                
        elif action == 4:#let turn
            if self.direction == -1:
                self.held_keys = ["w","c","d"]
                self.drift_held += .5
            elif self.direction == 0:
                self.held_keys = ["w","c","a"]
                self.drift_held += 1.05
                self.direction = -1
            else:
                self.held_keys = ["w"]
                self.direction = 0
                self.end_drift()

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
    counts = 0
    while True:
        steps += 1

        if keyboard.is_pressed('p'):
            action = 1
        elif keyboard.is_pressed('u'):
            action = 2
        elif keyboard.is_pressed('o'):
            action = 3
        elif keyboard.is_pressed('i'):
            action = 4
        else:
            action = 0

        
        state,reward,terminal,info = env.step(action)
        #time.sleep(0.025)
        """if steps % 500 == 0:
            print("Frames per Hour:")
            time_seg = 500 / (time.time() - starter)
            
            print(time_seg * 3600)
            starter = time.time()"""
        """
        Remember this wont work due to no delay!
        """
        
        score += reward
        #print(reward)
        if reward != 0:
            pass
            #print(reward)
            #print(counts)
            #counts = 0
        else:
            counts += 1
        #print(reward)
        if terminal:
            print("Total Reward: " + str(score))
            score = 0
            env.reset()
