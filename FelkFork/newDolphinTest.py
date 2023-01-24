from dolphin import event, gui
import sys
sys.path.append("C:\\Users\\TYLER\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages")
#import numpy as np
from PIL import Image
#import cv2
import random
import time

def show_screenshot(width: int, height: int, data: bytes):
    #print(f"received {width}x{height} image of length {len(data)}")
    # data is RGBA, so its size is width*height*4
    gui.draw_text((10, 50), red, f"Hi")

red = 0xffff0000
frame_counter = 0
start = time.time()
count = 0
while True:
    (width,height,data) = await event.framedrawn()
    gui.draw_text((10, 50), red, f"Hi")

    if random.random() > 0.995:
        image = Image.frombytes('RGBA', (width,height), data, 'raw')
        image.show()
"""advance = False
while True:
    await event.on_framedrawn(show_screenshot)
    count += 1
    gui.draw_text((10, 50), red, f"Count: {count}")"""

#image = Image.frombytes('RGBA', (width,height), data, 'raw')
#gui.draw_text((10, 50), red, f"Hi")




"""while True:
    #while not advance:
    #await event.frameadvance()
    
    advance = False
    
    await event.on_framedrawn(show_screenshot)
    

    #with open('loggTest.txt', 'a') as f:
        #f.write("Run framedrawn statement")
    
    frame_counter += 1
    fps = frame_counter / (time.time() - start)
    # draw on screen
    counts = count / (time.time() - start)
    gui.draw_text((10, 10), red, f"FPS: {fps}")
    gui.draw_text((10, 50), red, f"Count: {counts}")
    #gui.draw_text((10, 50), red, f"frames: {img.dtype}")
    # print to console
    if frame_counter % 60 == 0:
        print(f"The frame count has reached {frame_counter}")"""


"""global count
global advance
global img

gui.draw_text((10, 50), red, f"Boo")

if count % 4 == 3:
    image = Image.frombytes('RGBA', (width,height), data, 'raw')
    image = image.resize((94,78))
    image = image.convert("RGB")

    img1 = np.asarray(image)
    img1 = img1[...,::-1]
    img1 = np.dot(img1[...,:3], [0.2989, 0.5870, 0.1140])

    #if random.random() > 0.99:
        #cv2.imwrite("filename.png", img)

    img = img1.astype(np.float32)
    
    advance = True

count += 1"""
