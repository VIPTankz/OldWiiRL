
#Window is 500x270 when captured

try:
    import sys
    #sys.path.append("C:\\Users\\TYLER\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages")
    sys.path.append("/home/tyler/anaconda3/envs/effzero/lib/python3.8/site-packages")
except Exception as e:
    with open('logg.txt', 'a') as f:
        f.write(str(e))
    raise Exception("stop")

import os
cwd = str(os.getcwd())
cwd = cwd.split("dolphin",1)[1][0]
os.chdir('/home/tyler/Documents/EfficientZero')

with open('logg.txt', 'a') as f:
    f.write('Path Changed... Again')

with open('logg.txt', 'a') as f:
    f.write("PID:" + str(cwd))

pid = int(cwd)

from multiprocessing import shared_memory,Lock
import numpy as np

with open('logg.txt', 'a') as f:
    f.write('half libraries installed\n')

from PIL import Image
import math
import time
import random

with open('logg.txt', 'a') as f:
    f.write('Imported Some\n')

with open('logg.txt', 'a') as f:
    f.write('Got Pid ' + str(pid) + '\n')
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
    with open('logg.txt', 'a') as f:
        f.write('Joined Shared Memory')
    
except Exception as e:
    with open('logg.txt', 'a') as f:
        f.write(str(e))

    with open('logg.txt', 'a') as f:
        f.write(' Failed to create Shared Memory')

    raise Exception("Stop - failed to create shared mem")

#import mss
#import dxcam

from dolphin import event, gui,savestate,memory,controller
with open('logg.txt', 'a') as f:
    f.write('\nImported FelkLibs')

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

        with open('logg.txt', 'a') as f:
            f.write('Init Reset Successful\n')

    def reset(self):

        self.current_step = 0

        ########### Game Code
        self.movement_inc = 0.015
        x = np.random.random()
        global change
        change = False
        savestate.load_from_slot(1)#random.randint(1,8)
        
        """if x < 0.5:
            savestate.load_from_slot(1)
        elif x < 0.6:
            savestate.load_from_slot(5)
        elif x < 0.75:
            savestate.load_from_slot(4)
        elif x < 0.95:
            savestate.load_from_slot(2)
        else:
            savestate.load_from_slot(3)"""
        #else:
            #savestate.load_from_slot(4)
        
        change = True

        self.numEnemies = memory.read_u32(0x91CFA9E8)
        self.numLives = memory.read_u32(0x91D27ED0)
        self.x = 0
        self.y = 0
        
        ##################### End Game Code
        start = time.time()
        while True:
            time.sleep(0.5)

            if time.time() - start > 10:
                time.sleep(10)
                with open('logg.txt', 'a') as f:
                    f.write("Waiting 10+ seconds! PID: " + str(pid))
                    f.write('\nWaiting for Reset... ' + str(self.shm_array[0]) + " " + str(self.shm_array[1]) + " " + str(self.timestep))
                    f.write("\n\n")
                #self.timestep = 0
                #self.shm_array[0][0 + self.offset] = self.timestep
                start = time.time()
                
            if self.shm_array[0][1 + self.offset] == self.timestep:
                break

        #write state
        self.shm_array[0][4 + self.offset] = 0.
        self.shm_array[0][3 + self.offset] = 0.
        
        self.shm_array[1:] = np.zeros((Ymem,Xmem),dtype=np.float32)#self.get_state()

        self.timestep += 1
        self.shm_array[0][0 + self.offset] = self.timestep

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
        # Returns reward,terminal,trun
        controller.set_wii_buttons(0,self.dic)
        self.current_step += 1

        ########### Game Code
        terminal = False
        reward = 0.

        numEnemies = memory.read_u32(0x91CFA9E8)
        numLives = memory.read_u32(0x91D27ED0)

        #check if we died
        if numLives < self.numLives:
            return -1., True, False


        #check if the round ended
        if numEnemies > self.numEnemies:
            return 1., True, False

        #get kills
        reward = self.numEnemies - numEnemies

        self.numEnemies = numEnemies
        self.numLives = numLives

        ##################### End Game Code

        #remove this
        if random.randint(1,60) == 25:
            terminal = True

        return reward,terminal,False

    def apply_action(self,action):        

        """
        self.dic = {"Left":False,"Right":False,"Down":False,"Up":False, \
               "Plus":False,"Minus":False,"One":False,"Two":False, \
               "A":False,"B":False,"Home":False}

        """
        self.last_action = action
        self.dic = {"Left":False,"Right":False,"Down":False,"Up":False, \
               "Plus":False,"Minus":False,"One":False,"Two":False, \
               "A":False,"B":False,"Home":False}

        #REMOVE THIS LINE
        action = random.randint(0,8)

        if action == 0:
            self.dic["Left"] = True
        elif action == 1:
            self.dic["Right"] = True
        elif action == 2:
            self.dic["Up"] = True
        elif action == 3:
            self.dic["Down"] = True
        elif action == 4:
            self.x += self.movement_inc
        elif action == 5:
            self.x -= self.movement_inc
        elif action == 6:
            self.y += self.movement_inc
        elif action == 7:
            self.y -= self.movement_inc
        elif action == 8:
            self.dic["B"] = True

        self.x = max(-0.32,min(self.x,0.32))
        self.y = max(-0.16, min(self.y, 0.08))

        controller.set_wii_ircamera_transform(0,self.x,self.y,-2,0,0,0)
        controller.set_wii_buttons(0,self.dic)
        
    def step(self):
        
        #get action
        #sync
        while True:
            start = time.time()
            time.sleep(0.001)
            """with open('logg.txt', 'a') as f:
                f.write('\nWaiting for ETimestep... ' + str(self.shm_array[0]) + " " + str(self.shm_array[1]) + " " + str(self.timestep))"""

            if self.shm_array[0][1 + self.offset] == self.timestep:
                break

            if time.time() - start > 10:
                with open('logg.txt', 'a') as f:
                    f.write('\nDolphin has been awaiting respose for 10+ seconds! Pid: ' + str(pid))
                    f.write('\nWaiting for Step... ' + str(self.shm_array[0]) + " " + str(self.shm_array[1]) + " " + str(self.timestep))
                    f.write("\n\n")

        try:
            self.apply_action(self.shm_array[0][2 + self.offset])
        except:
            print("Error at apply action! PID: " + str(pid))
            time.sleep(1)


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

with open('logg.txt', 'a') as f:
    f.write('\nEntering Main While loop')

while True:
    env.step()
        
    for i in range(env.frameskip):
        (width,height,data) = await event.framedrawn()

        rewardN,terminalN,trunN = env.get_reward_terminal()
        #env.apply_action(env.last_action)
        gui.draw_text((10, 90), red, str(env.last_action))

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

    gui.draw_text((10, 10), red, f"HI")
    
    reward = 0
    terminal = False
    trun = False
        

