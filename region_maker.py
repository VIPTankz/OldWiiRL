import pygame
import pickle
import sys
from Region import Region
import math
from ButtonLib import Button

#run in 1920x1440
#When implementing use modulo not search!

#remember you need to specify num chkps in other file to handle looping

#todo
#add bound buttons
#add load/save buttons
#test load/save




#implement into main!

class RegionMaker():
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((1820,1340))
        all_fonts = pygame.font.get_fonts()
        self.font = pygame.font.SysFont(all_fonts[7], 45)
        self.top_font = pygame.font.SysFont(all_fonts[7], 30)
        self.fps = 30
        self.running = True
        self.mouse_up = False

        self.map = pygame.image.load("blank_regions.jpg").convert() #950x1220

        self.image_x = 950
        self.image_y = 1220

        self.start_x = 50
        self.start_y = 50

        self.grid_size = 10
        self.grid_x = int(self.image_x / self.grid_size)
        self.grid_y = int(self.image_y / self.grid_size)

        self.create_regions()
        
        self.mode = "bounds"
        self.shift_held = False
        self.held_key = None
        self.mouse_held = False
        self.drawing = False

        self.dir_num = 1
        self.chkp_num = 0
        self.in_bounds = False
        self.brush_size = 2

        
        self.define_buttons()
        self.define_chkp_buttons()
        self.define_dir_buttons()
        self.define_brush_buttons()
        self.define_bound_buttons()
        self.define_saving_buttons()

    def save(self):
        save_name = "regions.dat"
        with open(save_name, "wb") as f:
            pickle.dump(self.regions, f)
        print("Saved")

    def load(self):
        save_name = "regions.dat"
        with open(save_name, "rb") as f:
            self.regions = pickle.load(f)
        print("Loaded")

    def define_saving_buttons(self):
        self.save_button = Button(1100,50,200,75,(128,128,128),(0,255,0),text = "Save")
        self.load_button = Button(1500,50,200,75,(128,128,128),(0,255,0),text = "Load")

    def define_bound_buttons(self):
        self.bound_buttons = []
        texts = ["Inbound","Outbound"]
        for i in range(2):
            if i == 0:
                if self.in_bounds:
                    self.bound_buttons.append(Button(1100,350,200,50,(0,128,0),(0,255,0),text = texts[i]))
                else:
                    self.bound_buttons.append(Button(1100,350,200,50,(128,128,128),(0,255,0),text = texts[i]))
            else:
                if self.in_bounds:
                    self.bound_buttons.append(Button(1100,400,200,50,(128,128,128),(0,255,0),text = texts[i]))
                else:
                    self.bound_buttons.append(Button(1100,400,200,50,(0,128,0),(0,255,0),text = texts[i]))                     

    def define_brush_buttons(self):
        self.brush_buttons = []
        for i in range(3):
            i += 1
            
            if self.brush_size == i:
                self.brush_buttons.append(Button(1500,175 + i * 25,200,25,(0,128,0),(0,255,0),text = "Brush:" + str(i)))
            else:
                self.brush_buttons.append(Button(1500,175 + i * 25,200,25 ,(128,128,128),(0,255,0),text = "Brush:" + str(i)))
            

    def define_dir_buttons(self):
        self.dir_buttons = []
        for i in range(7):
            i -= 3

            if self.dir_num == i:
                self.dir_buttons.append(Button(1200,675 + i * 25,150,25,(0,128,0),(0,255,0),text = str(i)))
            else:
                self.dir_buttons.append(Button(1200,675 + i * 25,150,25 ,(128,128,128),(0,255,0),text = str(i)))
            
    def define_chkp_buttons(self):
        self.chkp_buttons = []
        for i in range(31):
            i -= 1
            if self.chkp_num == i:
                self.chkp_buttons.append(Button(1550,500 + i * 25,150,25,(0,128,0),(0,255,0),text = str(i)))
            else:
                self.chkp_buttons.append(Button(1550,500 + i * 25,150,25,(128,128,128),(0,255,0),text = str(i)))

    def define_buttons(self):
        button_width = 420
        self.bound_button = Button(1050,200,button_width,100,(128,128,128),(0,255,0),text = "Bounds")
        self.x_button = Button(1050,500,button_width,100,(128,128,128),(0,255,0),text = "X Direction (green is right)")
        self.y_button = Button(1050,800,button_width,100,(128,128,128),(0,255,0),text = "Y Direction (green is up)")
        self.chkp_button = Button(1050,1100,button_width,100,(128,128,128),(0,255,0),text = "Checkpoints")

        if self.mode == "bounds":
            self.bound_button = Button(1050,200,button_width,100,(0,128,0),(0,255,0),text = "Bounds")
        elif self.mode == "x":
            self.x_button = Button(1050,500,button_width,100,(0,128,0),(0,255,0),text = "X Direction (green is right)")
        elif self.mode == "y":
            self.y_button = Button(1050,800,button_width,100,(0,128,0),(0,255,0),text = "Y Direction (green is up)")
        elif self.mode == "chkp":
            self.chkp_button = Button(1050,1100,button_width,100,(0,128,0),(0,255,0),text = "Checkpoints")
             

    def create_regions(self):
        self.regions = []
        for j in range(self.grid_y):
            for i in range(self.grid_x):
                self.regions.append(Region(self.grid_size*i,self.grid_size*j,
                                           self.grid_size*(i + 1)-1,self.grid_size*(j+1)-1,
                                           0,0,
                                           True,False,-1))

    def show_bounds(self):
        for i in self.regions:
            if not i.in_bounds:
                s = pygame.Surface((self.grid_size,self.grid_size))  # the size of your rect
                s.set_alpha(90)                # alpha level
                s.fill((128,20,20))           # this fills the entire surface
                self.screen.blit(s, (self.start_x + i.x,self.start_y + i.y))

    def show_x(self):
        for i in self.regions:
            if i.dir_x > 0.1:
                s = pygame.Surface((self.grid_size,self.grid_size))  # the size of your rect
                s.set_alpha(120)                # alpha level
                s.fill((0,85 * i.dir_x,0))           # this fills the entire surface
                self.screen.blit(s, (self.start_x + i.x,self.start_y + i.y))
            elif i.dir_x < -0.1:
                s = pygame.Surface((self.grid_size,self.grid_size))  # the size of your rect
                s.set_alpha(120)                # alpha level
                s.fill((85 * -i.dir_x,0,85 * -i.dir_x))           # this fills the entire surface
                self.screen.blit(s, (self.start_x + i.x,self.start_y + i.y))                

    def show_y(self):
        for i in self.regions:
            if i.dir_y < -0.1:
                s = pygame.Surface((self.grid_size,self.grid_size))  # the size of your rect
                s.set_alpha(120)                # alpha level
                s.fill((0,85 * -i.dir_y,0))           # this fills the entire surface
                self.screen.blit(s, (self.start_x + i.x,self.start_y + i.y))
            elif i.dir_y > 0.1:
                s = pygame.Surface((self.grid_size,self.grid_size))  # the size of your rect
                s.set_alpha(120)                # alpha level
                s.fill((85 * i.dir_y,0,85 * i.dir_y))           # this fills the entire surface
                self.screen.blit(s, (self.start_x + i.x,self.start_y + i.y))

    def show_chkp(self):
        for i in self.regions:
            if i.is_chkp:
                s = pygame.Surface((self.grid_size,self.grid_size))  # the size of your rect
                s.set_alpha(120)                # alpha level
                
                s.fill(self.get_chkp_col(i.chkp_num))           # this fills the entire surface
                self.screen.blit(s, (self.start_x + i.x,self.start_y + i.y))#(220 - (i.chkp_num % 10) * 12,220 - (i.chkp_num) * 12,0)

    def get_chkp_col(self,num):
        
        #num is chkp number
        if num % 8 == 0:
            return (176,23,31)
        elif num % 8 == 1:
            return (0,0,255)
        elif num % 8 == 2:
            return (0,255,0)
        elif num % 8 == 3:
            return (255,255,0)
        elif num % 8 == 4:
            return (238,118,33)
        elif num % 8 == 5:
            return (255,0,0)
        elif num % 8 == 6:
            return (0,245,255)
        elif num % 8 == 7:
            return (255,0,255)


    def draw_grid(self):
        #vertical lines
        for i in range(self.grid_x + 1):
            pygame.draw.line(self.screen, (255,255,255),
                    (self.start_x + self.grid_size * i, self.start_y), (self.start_x + self.grid_size * i, self.start_y + self.image_y))

        #horizontal lines
        for i in range(self.grid_y + 1):
            pygame.draw.line(self.screen, (255,255,255),
                    (self.start_x, self.start_y + self.grid_size * i), (self.start_x + self.image_x, self.start_y + self.grid_size * i))

    def process_clicks(self,mouse_pos):
        if self.drawing:
            if mouse_pos[0] >= self.start_x and mouse_pos[1] >= self.start_y \
               and mouse_pos[0] <= self.start_x + self.image_x and mouse_pos[1] <= self.start_y + self.image_y:
                
                x = mouse_pos[0] - self.start_x
                y = mouse_pos[1] - self.start_y

                x = math.floor(x / self.grid_size)
                y = math.floor(y / self.grid_size)

                num = self.convert_xy_to_num(x,y)

                if self.shift_held:
                    self.held_key = num
                elif self.held_key != None:
                    nums = self.get_box(num)
                    self.held_key = None
                    for num in nums:
                        self.toggle(num)
                    
                else:
                    nums = self.get_brush_nums(num,x,y)
                    for num in nums:
                        self.toggle(num)
            else:
                self.held_key = None

    def get_brush_nums(self,num,x,y):
        nums = [num]
        if self.brush_size > 1:
            if x + 1 < self.grid_x:
                nums.append(self.convert_xy_to_num(x + 1,y))
            if x - 1 > -1:
                nums.append(self.convert_xy_to_num(x - 1,y))
            if y - 1 > -1:
                nums.append(self.convert_xy_to_num(x,y - 1))
            if y + 1 < self.grid_y:
                nums.append(self.convert_xy_to_num(x,y + 1))
        if self.brush_size > 2:
            if x + 2 < self.grid_x:
                nums.append(self.convert_xy_to_num(x + 2,y))
            if x - 2 > -1:
                nums.append(self.convert_xy_to_num(x - 2,y))
            if y - 2 > -1:
                nums.append(self.convert_xy_to_num(x,y - 2))
            if y + 2 < self.grid_y:
                nums.append(self.convert_xy_to_num(x,y + 2))

            if x + 1 < self.grid_x and y + 1 < self.grid_y:
                nums.append(self.convert_xy_to_num(x + 1,y + 1))
            if x - 1 > -1 and y + 1 < self.grid_y:
                nums.append(self.convert_xy_to_num(x - 1,y + 1))
            if x + 1 < self.grid_x and y - 1 > -1:
                nums.append(self.convert_xy_to_num(x + 1,y - 1))
            if x - 1 > -1 and y - 1 > -1:
                nums.append(self.convert_xy_to_num(x - 1,y - 1))

        return nums

    def get_box(self,num):
        #we want point between num and self.held_key
        p1x,p1y = self.convert_num_to_xy(num)
        p2x,p2y = self.convert_num_to_xy(self.held_key)

        if p1x >= p2x:
            startx = p2x
        else:
            startx = p1x

        difx = abs(p1x - p2x) + 1

        if p1y >= p2y:
            starty = p1y
        else:
            starty = p2y

        dify = abs(p1y - p2y) + 1

        nums = []
        for i in range(difx):
            for j in range(dify):
                nums.append(self.convert_xy_to_num(startx + i,starty - j))
                        
        return nums

    def convert_xy_to_num(self,x,y):
        return x + y * self.grid_x

    def convert_num_to_xy(self,num):
        x = num % self.grid_x
        y = math.floor(num / self.grid_x)

        return x,y

    def toggle(self,num):
        if self.mode == "bounds":
            self.regions[num].in_bounds = self.in_bounds
        elif self.mode == "x":
            self.regions[num].dir_x = self.dir_num
        elif self.mode == "y":
            self.regions[num].dir_y = self.dir_num
        elif self.mode == "chkp":
            self.regions[num].chkp_num = self.chkp_num
            if self.chkp_num == -1:
                self.regions[num].is_chkp = False
            else:
                self.regions[num].is_chkp = True               
            

    def update(self):
        
        self.screen.fill((0,0,0))
        self.clock.tick(self.fps)

        
        self.mouse_up = False
        #allow shutdown window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.mouse_up = True
                self.drawing = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.drawing = True
            

        mouse_pos = pygame.mouse.get_pos()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT]:
            self.shift_held = True
        else:
            self.shift_held = False


        self.screen.blit(self.map,(self.start_x,self.start_y))

        if self.mode == "bounds":
            self.show_bounds()
        elif self.mode == "x":
            self.show_x()
        elif self.mode == "y":
            self.show_y()
        elif self.mode == "chkp":
            self.show_chkp()
        
        self.draw_grid()

        self.process_clicks(mouse_pos)

        self.bound_button.create(self.screen)
        self.x_button.create(self.screen)
        self.y_button.create(self.screen)
        self.chkp_button.create(self.screen)

        if self.bound_button.click(mouse_pos,self.mouse_up):
            self.mode = "bounds"
            self.define_buttons()
        elif self.x_button.click(mouse_pos,self.mouse_up):
            self.mode = "x"
            self.define_buttons()
        elif self.y_button.click(mouse_pos,self.mouse_up):
            self.mode = "y"
            self.define_buttons()
        elif self.chkp_button.click(mouse_pos,self.mouse_up):
            self.mode = "chkp"
            self.define_buttons()

        for i in range(len(self.dir_buttons)):
            self.dir_buttons[i].create(self.screen)
            if self.dir_buttons[i].click(mouse_pos,self.mouse_up):
                self.dir_num = i - 3
                self.define_dir_buttons()

        for i in range(len(self.chkp_buttons)):
            self.chkp_buttons[i].create(self.screen)
            if self.chkp_buttons[i].click(mouse_pos,self.mouse_up):
                self.chkp_num = i - 1
                self.define_chkp_buttons()

        for i in range(len(self.brush_buttons)):
            self.brush_buttons[i].create(self.screen)
            if self.brush_buttons[i].click(mouse_pos,self.mouse_up):
                self.brush_size = i + 1
                self.define_brush_buttons()

        for i in range(len(self.bound_buttons)):
            self.bound_buttons[i].create(self.screen)
            if self.bound_buttons[i].click(mouse_pos,self.mouse_up):
                if i == 0:
                    self.in_bounds = True
                else:
                    self.in_bounds = False
                self.define_bound_buttons()

        self.save_button.create(self.screen)
        self.load_button.create(self.screen)
        if self.save_button.click(mouse_pos,self.mouse_up):
            self.save()
        elif self.load_button.click(mouse_pos,self.mouse_up):
            self.load()

        pygame.display.flip()
        return self.running

if __name__ == "__main__":
    
    reg = RegionMaker()

    running = True
    while running:
        running = reg.update()
