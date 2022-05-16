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
    "d": 0x20
    }

#32400 frames/hour!
#ray did 28700
class MarioKartEnv():
    def __init__(self,config=None):

        windows = Desktop(backend="uia").windows()
        for i in windows:
            if i.window_text()[:19] == "Dolphin 5.0-16101 |":
                window_name = i.window_text()

        self.hwnd = win32gui.FindWindow(None, window_name)
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        self.w = right - left
        self.h = bot - top

        self.template = cv2.imread('C:/Users/TYLER/Downloads/dolphin_ai_tests/env/funky_kong_img2.png')
        self.template = cv2.cvtColor(self.template, cv2.COLOR_RGB2GRAY)
        self.tem_w = 69
        self.tem_h = 132#100,141

        self.action_space = gym.spaces.Discrete(4)
        """
        
        1 - accel
        2 - accel+wheely
        3 - accel+drift_hold_right
        4 - accel+drift_hold_left


        item has been removed
        5 - accel + right
        6 - accel + left
        7 - accel + item
        0 - null
        """
        #yx
        #self.observation_space = gym.spaces.Box(
        #low=0, high=255, shape=(64, 32), dtype=np.uint8)

        save_name = "regions.dat"

        self.image_x = 950
        self.image_y = 1220
        self.grid_size = 10
        self.grid_x = int(self.image_x / self.grid_size)
        self.grid_y = int(self.image_y / self.grid_size)
        
        self.method = eval('cv2.TM_CCOEFF')
        self.num_chkps = 22
        with open(save_name, "rb") as f:
            self.regions = pickle.load(f)
        
        self.reset()


    def reset(self):
        self.dist = 0
        self.first = True
        release_keys()
        self.held_keys = []
        KeyPress("m")
        time.sleep(0.25)
        self.timer = time.time()
        self.prev_action = 0
        self.out_frames = 0
        self.current_chkp = -1

        return self.get_state()[0]

    def template_match(self,img):
        terminal = False
        #crop image so avoid issues -- #og image 2098, 3868
        img = img[680:1900, 2600: 3550]
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        # Apply template Matching
        res = cv2.matchTemplate(img,self.template,self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

        if not self.first:
            
            self.prev_top_left = copy(self.top_left)

        self.top_left = max_loc

        if self.first:
            self.prev_top_left = copy(self.top_left)
            self.first = False
            #cv2.imwrite("bug_test" + str(time.time()) + ".jpg", img)
            return 0,False
            
        else:

            y_dif = list(self.prev_top_left)[1] - list(self.top_left)[1]
            x_dif = list(self.top_left)[0] - list(self.prev_top_left)[0]

        if True:#time.time() - self.timer > 5.4
        
            self.dist = x_dif**2 + y_dif**2

            #exception for broken template matching
            if self.dist > 900:

                #need to allow it to refind template next frame
                self.first = True
                reward = 0.9
                bottom_right2 = (self.prev_top_left[0] + self.tem_w, self.prev_top_left[1] + self.tem_h)
                cv2.rectangle(img,self.prev_top_left, bottom_right2, 255, 2)
                
                bottom_right = (self.top_left[0] + self.tem_w, self.top_left[1] + self.tem_h)
                cv2.rectangle(img,self.top_left, bottom_right, 255, 2)
                
                cv2.imwrite("wrong_pattern12" + str(round(time.time(),4)) + ".jpg", img)

                self.dist = 0
            else:

                #region code - #cropped image 1220, 880 (y,x)
                reward = self.get_reward(x_dif,y_dif)
                if self.out_frames > 3:
                    terminal = True
                    reward -= 150

                
                """bottom_right = (self.top_left[0] + self.tem_w, self.top_left[1] + self.tem_h)
                cv2.rectangle(img,self.top_left, bottom_right, 128, 2)
                cv2.imwrite("bug_test" + str(time.time()) + ".jpg", img)

                raise Exception("stop")"""
                
        else:
            return 0,terminal

        reward = reward / 30
        reward -= 0.03

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

        x = math.floor(x / self.grid_size)
        y = math.floor(y / self.grid_size)

        #get which grid cell
        num = self.convert_xy_to_num(x,y)

        #check out of bounds
        if not self.regions[num].in_bounds:
            self.out_frames += 1
            reset_frames = False

        #check dir_x
        reward += x_dif * self.regions[num].dir_x

        #check dir_y
        reward -= y_dif * self.regions[num].dir_y

        #checkpoints
        if self.regions[num].is_chkp:
            if self.regions[num].chkp_num > self.current_chkp or (self.regions[num].chkp_num == 0 and self.current_chkp == self.num_chkps):
                reward += 65
                #print("checkpoint: " + str(self.regions[num].chkp_num))
                self.current_chkp = self.regions[num].chkp_num
            elif self.regions[num].chkp_num < self.current_chkp or \
                 (self.regions[num].chkp_num == self.num_chkps and (self.current_chkp == 0 or self.current_chkp == -1)):
                
                self.out_frames += 1
                reset_frames = False        
        
        if reset_frames:
            self.out_frames = 0
            
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
        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, self.w, self.h)

        saveDC.SelectObject(saveBitMap)

        # Change the line below depending on whether you want the whole window
        # or just the client area. 
        #result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
        result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 0)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)#

        #og image 2098, 3868
        im = np.array(im)
        
        #gets the top_left var
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        reward,terminal = self.template_match(im)
        
        
        im = im[80:, 85: 3868 - 85]
        
        im = cv2.resize(im, (64,32), interpolation = cv2.INTER_AREA)
        #cv2.imwrite("bug_test_ai" + str(time.time()) + ".jpg", im)
        #raise Exception("stop")

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        return im,reward,terminal

    def step(self,action=0):
        #time.sleep(0.005)
        terminal = False

        self.apply_action(action)
        #press some key
        state,reward,terminal = self.get_state()

        if time.time() - self.timer > 80:
            terminal = True

        #print(reward)
        info = {}

        return state,reward,terminal,info

        

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
        if action == 1:
            self.held_keys = ["w"]
        elif action == 2:
            self.held_keys = ["w","e"]
        elif action == 3:
            self.held_keys = ["w","c","d"]
        elif action == 4:
            self.held_keys = ["w","c","a"]
        """elif action == 5:
            self.held_keys = ["w","d"]
        elif action == 6:
            self.held_keys = ["w","a"]
        elif action == 7:
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
    
    while True:

        if keyboard.is_pressed('u'):
            action = 0
        elif keyboard.is_pressed('h'):
            action = 3
        elif keyboard.is_pressed('k'):
            action = 2
        elif keyboard.is_pressed('i'):
            action = 1
        else:
            action = 0

            
        state,reward,terminal,info = env.step(action)
        score += reward
        print(reward)
        if terminal:
            print("Total Reward: " + str(score))
            score = 0
            env.reset()
