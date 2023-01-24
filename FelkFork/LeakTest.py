import sys
sys.path.append("C:\\Users\\TYLER\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages")
with open('leak.txt', 'w') as f:
    f.write("got path123")
from PIL import Image

from dolphin import event,gui

from copy import deepcopy

import numpy as np
import time
with open('leak.txt', 'a') as f:
    f.write("\nlibraries")
    
white = 0xffffffff

def show_screenshot(width: int, height: int, data: bytes):
    global allow
    allow = True
    #gui.draw_text((10, 10), white, "Hi")
    #print(f"received {width}x{height} image of length {len(data)}")
    #image = Image.frombytes('RGBA', (width,height), data, 'raw')
    #image.show()

steps = 1
start = time.time()
allow = False
while True:
    await event.frameadvance()
    steps += 1
    fps = round(steps / (time.time() - start))
    gui.draw_text((10, 10), white, "FPS: " + str(fps))

    
"""event.on_framedrawn(show_screenshot)
with open('leak.txt', 'a') as f:
    f.write("\nonframeadvance")
while True:

    while not allow:
        await event.frameadvance()
    
    with open('leak.txt', 'a') as f:
        f.write("\nallowed")
        
    (width,height,data) = await event.framedrawn()
    with open('leak.txt', 'a') as f:
        f.write("\ndrawn")
    allow = False
    #Adding the two lines below fixes the problem?!?
    
    #with open('leak.txt', 'a') as f:
    #pass
    
    #img = Image.frombytes('RGBA', (width,height), data, 'raw')"""


