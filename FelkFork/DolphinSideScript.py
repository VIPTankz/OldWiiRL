
#Window is 500x270 when captured

with open('logg.txt', 'w') as f:
    f.write('started')
try:
    import sys
    sys.path.append("C:\\Users\\TYLER\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages")
except Exception as e:
    with open('logg.txt', 'w') as f:
        f.write(str(e))

    raise Exception("stop")

with open('logg.txt', 'w') as f:
    f.write('Path Changed... Again')

from multiprocessing import shared_memory,Lock
import numpy as np
import time
import random
from copy import deepcopy
import cv2
from PIL import Image
import math

#lock = Lock()

"""with open('logg.txt', 'w') as f:
    f.write('Imported Some')"""

with open('pid_num.txt') as f:
    pid = int(f.readlines()[0])


with open('logg.txt', 'w') as f:
    f.write('Got Pid')
#Ymem = 270
#Xmem = 500
#Ymem = 108
#Xmem = 200

Ymem = 78
Xmem = 94

##78x94

#div by 6.8

try:
    data = np.zeros((Ymem + 1,Xmem),dtype=np.float32)
    shm = shared_memory.SharedMemory(name='p' + str(pid))
    with open('logg.txt', 'w') as f:
        f.write('Made Shared Memory')
    
except Exception as e:
    with open('logg.txt', 'a') as f:
        f.write(str(e))

    with open('logg.txt', 'w') as f:
        f.write('Failed to create Shared Memory')

#import mss
#import dxcam

from dolphin import event, gui,savestate,memory,controller
with open('logg.txt', 'a') as f:
    f.write('Create Shared3')

class DolphinSideEnv():
    def __init__(self,pid=0,offset = 0):

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

        #about 78fs with mss

        #about 60 with dxcam (the weird method)

        ########### Game Code

        self.last_action = 0
        
        ##################### End Game Code

        self.offset = offset
        
        self.window_header = 40
        self.window_width = 100
        self.window_height = 60

        pidx = pid % 5
        pidy = math.floor(pid / 5)
        
        #self.monitor = {"top": 32 + (270 + 32) * pidy, "left": pidx*500, "width": 500, "height": 270}
        #self.monitor = {"top": 32 + (Ymem + 32) * pidy, "left": pidx*(Xmem + 1), "width": Xmem, "height": Ymem}

        self.frameskip = 4

        self.timestep = 0.

        self.current_step = 0

        with open('logg.txt', 'a') as f:
            f.write('About to make data array\n')        

        self.data = np.zeros((Ymem + 1,Xmem),dtype=np.float32)#np.zeros(self.dims,dtype=np.float32)

        self.shm_array = np.ndarray(self.data.shape, dtype=self.data.dtype, buffer=shm.buf)

        with open('logg.txt', 'a') as f:
            f.write('shared mem\n')

        self.reset()

        #with open('logg.txt', 'a') as f:
            #f.write('Reset\n')

    def reset(self):

        self.current_step = 0

        ########### Game Code
        x = np.random.random()
        global change
        change = False
        #savestate.load_from_slot(1)
        
        if x < 0.5:
            savestate.load_from_slot(1)
        elif x < 0.6:
            savestate.load_from_slot(5)
        elif x < 0.75:
            savestate.load_from_slot(4)
        elif x < 0.95:
            savestate.load_from_slot(2)
        else:
            savestate.load_from_slot(3)
        #else:
            #savestate.load_from_slot(4)
        
        change = True
            
        #self.mHealth = memory.read_u16(0x8060D1D2)
        self.bHealth = memory.read_u16(0x816C7F3A)
        self.bDist = memory.read_f32(0x81541F8C)
        #self.finished = memory.read_u16(0x8060E034)
        self.height = memory.read_u16(0x8127F452)
        self.timer = memory.read_f32(0x81541B70)
        self.bat = memory.read_u16(0x8061ABF6)

        #self.mLives = memory.read_u8(0x8060CB1F)
        #self.bLives = memory.read_u8(0x8060CB23)
        
        ##################### End Game Code
                
        while True:
            #time.sleep(0.2)
            #with open('logg.txt', 'a') as f:
                #f.write('\nWaiting for ETimestep... ' + str(self.shm_array[0]) + " " + str(self.shm_array[1]) + " " + str(self.timestep))
                
            if self.shm_array[0][1 + self.offset] == self.timestep:
                break

        #write state
        #lock.acquire()
        self.shm_array[0][4 + self.offset] = 0.
        self.shm_array[0][3 + self.offset] = 0.
        
        self.shm_array[1:] = np.zeros((Ymem,Xmem),dtype=np.float32)#self.get_state()

        self.timestep += 1
        self.shm_array[0][0 + self.offset] = self.timestep
        #lock.release()

        self.dic = {"Left":False,"Right":False,"Down":False,"Up":False, \
               "Plus":False,"Minus":False,"One":False,"Two":False, \
               "A":False,"B":False,"Home":False}

    def get_state(self):
        
        #event.on_framedrawn(show_screenshot)

        
        return img[:]

    def get_state_old(self):
        
        with mss.mss() as sct:
            
            # Part of the screen to capture
            #im = 0.07 * im[:,:,2] + 0.72 * im[:,:,1] + 0.21 * im[:,:,0]
            # Get raw pixels from the screen, save it to a Numpy array
            im = np.array(sct.grab(self.monitor))
            
            #im = 0.0002745098 * im[:,:,2] + 0.00282352941 * im[:,:,1] + 0.00082352941 * im[:,:,0]
            im = 0.07 * im[:,:,2] + 0.72 * im[:,:,1] + 0.21 * im[:,:,0]
            #im = im.astype(np.float32)

        return im

    def get_state_dx(self):

        im = self.camera.get_latest_frame()
        im = np.squeeze(im)
        im = np.true_divide(im,255,dtype=np.float32)
        #im = cv2.resize(im, dsize=(54, 100), interpolation=cv2.INTER_CUBIC)
        #im = np.swapaxes(im,0,1)

        return im

    def get_reward_terminal(self):
        
        controller.set_wii_buttons(0,self.dic)
        
        self.current_step += 1
        #Returns reward,terminal,trun
        ########### Game Code
        
        bHealth = memory.read_u16(0x816C7F3A)
        bDist = memory.read_f32(0x81541F8C)
        self.finished = memory.read_f32(0x81652A3C)
        self.height = memory.read_f32(0x8126B634)
        self.timer = memory.read_f32(0x81541B70)
        #bat = memory.read_u16(0x8061ABF6)

        #gui.draw_text((10, 10), 0xffff0000, f"Lives: {mLives}")
        #gui.draw_text((10, 50), 0xffff0000, f"percent: {mHealth}")

        reward = 0.
        terminal = 0.
        
        reward += (bHealth - self.bHealth) / 30
        reward += (bDist - self.bDist) / 150

        self.bHealth = bHealth
        self.bDist = bDist

        """if (bHealth - self.bHealth) / 30 > 0.01:

            with open('logg.txt', 'a') as f:
                f.write("Bhealth")
                f.write((bHealth - self.bHealth) / 30)
            
        if (bDist - self.bDist) / 125 > 0.01:

            with open('logg.txt', 'a') as f:
                f.write("bdist")
                f.write((bDist - self.bDist) / 125)"""

        if self.finished > 0:
            terminal = 1.

        if self.timer < 90 and self.timer > 0 and self.last_action == 11:
            reward += 0.0001
            """with open('logg.txt', 'a') as f:
                f.write("Last action: +0.001")"""

        if self.height < 0:
            return -1,1.,False

        """if self.bat != bat:
            reward += 1.
            with open('logg.txt', 'a') as f:
                f.write("Bat: +1")

        self.bat = bat"""

        return reward,terminal,False

        ##################### End Game Code        

    def apply_action(self,action):        

        """
        self.dic = {"Left":False,"Right":False,"Down":False,"Up":False, \
               "Plus":False,"Minus":False,"One":False,"Two":False, \
               "A":False,"B":False,"Home":False}

        -removed 1,2,6
        actions:
        noop 0
        left 1,right 2,up 3,down 4
        left 5,right 6,up 7,down 8 + One
        left9,right10,up11,down12 + Two
        left + One + Two 13
        right + One + Two 14
        One 15
        Two 16
        One + Two 17

        NEW ACTIONS
        0 - None
        1 - Move left
        2 - Move Right
        3 - One Left
        4 - One Right
        5 - One Down
        6 - Two Left
        7 - Two Right
        8 - Two Up
        9 - Two Down
        10 - SMASH LEFT
        11 - SMASH RIGHT
        12 - One
        13 - Two
        """

        self.dic = {"Left":False,"Right":False,"Down":False,"Up":False, \
               "Plus":False,"Minus":False,"One":False,"Two":False, \
               "A":False,"B":False,"Home":False}

        if action != self.last_action:
            pass
        else:
            if action == 5 or action == 9: #Down
                self.dic["Left"] = True
            elif action == 8: #Up
                self.dic["Right"] = True

            elif action == 1 or action == 3 or action == 6 or action == 10: #left
                self.dic["Up"] = True

            elif action == 2 or action == 4 or action == 7 or action == 11: #right
                self.dic["Down"] = True

            if action == 3 or action == 4 or action == 5 or action == 10 or action == 11 or action == 12:
                self.dic["One"] = True

            if action == 6 or action == 7 or action == 8 or action == 9 or action == 10 or action == 11 or action == 13:
                self.dic["Two"] = True
        
            
        self.last_action = action
            
        controller.set_wii_buttons(0,self.dic)
        
    def step(self):
        
        #get action
        #sync
        while True:
            #time.sleep(0.2)

            if self.shm_array[0][1 + self.offset] == self.timestep:
                break


        self.apply_action(self.shm_array[0][2 + self.offset])


    def step2(self,reward,terminal,trun,image):

        #with open('logg.txt', 'a') as f:
            #f.write('\nWriting timestep: ' + str(self.timestep))

        #image = cv2.resize(image,(78,94), interpolation=cv2.INTER_AREA)

        #send back data

        image = image.resize((94,78))
        image = image.convert("RGB")

        img1 = np.asarray(image)
        img1 = img1[...,::-1]
        image = np.dot(img1[...,:3], [0.2989, 0.5870, 0.1140])

        #if random.random() > 0.99:
            #cv2.imwrite("filename.png", img)

        image = image.astype(np.float32)
        
        self.shm_array[0][4 + self.offset] = float(terminal)
        self.shm_array[0][3 + self.offset] = reward
        self.shm_array[1:] = image#self.get_state()np.zeros((Ymem,Xmem),dtype=np.float32)#
        
        self.timestep += 1
        self.shm_array[0][0 + self.offset] = self.timestep

        if terminal or trun:
            self.reset()

"""def show_screenshot(width: int, height: int, data: bytes):
    #print(f"received {width}x{height} image of length {len(data)}")
    # data is RGBA, so its size is width*height*4
    
    if change:
        global img
        img = deepcopy(Image.frombytes('RGBA', (width,height), data, 'raw'))


    return"""
#img = np.zeros((528,640),dtype=np.float32)

#img = np.zeros((94,78),dtype=np.uint8)
change = True
#event.on_framedrawn(show_screenshot)

for i in range(4):
    await event.frameadvance()

env = DolphinSideEnv(pid=pid)

for i in range(env.frameskip):
    await event.frameadvance()

reward = 0
terminal = False
trun = False
red = 0xffff0000

while True:
    
    env.step()
        
    for i in range(env.frameskip):
        (width,height,data) = await event.framedrawn()
        with open('leak.txt', 'a') as f:
            pass

        rewardN,terminalN,trunN = env.get_reward_terminal()
        env.apply_action(env.last_action)

        #with open('logg.txt', 'a') as f:
            #f.write('\nAfter reward terminal')

        terminal = terminal or terminalN
        trun = trun or trunN
        reward += rewardN
        if terminal or trun:
            for i in range(4):
                await event.frameadvance()
            break

    #with open('logg.txt', 'a') as f:
        #f.write(str(deepcopy(img)))
    img = Image.frombytes('RGBA', (width,height), data, 'raw')
    
    env.step2(reward,terminal,trun,img)

    #gui.draw_text((10, 10), red, f"HI")
    
    reward = 0
    terminal = False
    trun = False
        

