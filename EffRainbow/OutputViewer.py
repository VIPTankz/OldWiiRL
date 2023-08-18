import pygame
import numpy as np
from copy import copy
import sys
import time

class OutputViewer():
    def __init__(self,tags):

        self.tags = tags

        self.width = 600
        self.height = 400

        self.text_height = 30

        self.bar_heights = self.height - (self.text_height * 2)

        self.rangeMin = -1
        self.rangeMax = 9
        self.remapping = [2, 0, 1, 3, 4]

        self.mult = self.height / self.rangeMax

        pygame.init()

        all_fonts = pygame.font.get_fonts()
        self.font = pygame.font.SysFont(all_fonts[7], 18)
        
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.color = (0, 0, 255)

    def update(self,ovals):
        time.sleep(0.01)

        vals = []
        for i in range(len(ovals)):
            vals.append(-1)

        for i in range(len(ovals)):
            vals[self.remapping[i]] = ovals[i]

        self.screen.fill((0,0,0))

        bar_width = (self.width - 80) / len(vals)
        spacing = bar_width / 10

        ma = np.argmax(vals)
        
        for i in range(len(vals)):
            if i == ma:
                color = (255,215,0)
            elif vals[i] < 0:
                color = (255,0,0)
            else:
                color = (0,0,255)
            
            vals[i] -= self.rangeMin

            pygame.draw.rect(self.screen, color, pygame.Rect(10 + spacing * i + bar_width * i,\
                                            self.bar_heights - int(vals[i] * self.mult), bar_width - spacing * 2, int(vals[i] * self.mult)))

            text = self.font.render(self.tags[i],1,(255,255,255))#creates the text
            self.screen.blit(text,(10 + spacing * i + bar_width * i,self.bar_heights + 10))

            text = self.font.render(str(round(vals[i],2)),1,(255,255,255))#creates the text
            self.screen.blit(text,(10 + spacing * i + bar_width * i,self.bar_heights + 30))   


        self.mouse_up = False
        #allow shutdown window
        self.mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.mouse_up = True

        self.clock.tick(60)

        pygame.display.flip()


if __name__ == "__main__":
    out = OutputViewer(["hLeft","sLeft","wLeft","Forward","wRight","sRight","hRight"])
    outputs = [1,2,3,4,5,6,12]
    while True:

        out.update(copy(outputs))
        for i in range(len(outputs)):
            outputs[i] += np.random.random() - 0.5
            if outputs[i] < -1:
                outputs[i] = -1
            elif outputs[i] > 12:
                outputs[i] = 12
    
