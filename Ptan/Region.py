

class Region():
    def __init__(self,x,y,bot_x,bot_y,
                 is_chkp,chkp_num = -1,focal = False):
        
        self.x = x
        self.y = y
        self.focal = focal

        self.bot_x = bot_x
        self.bot_y = bot_y

        #dir will be -1,0 or 1. This gives reward
        #self.dir_x = dir_x
        #self.dir_y = dir_y

        #self.in_bounds = in_bounds

        self.is_chkp = is_chkp
        self.chkp_num = chkp_num

    def in_region(self,x,y):
        #pass in the midpoint of funky kong

        if x >= self.x and y >= self.y:
            if x <= self.bot_x and y <= self.bot_y:
                return True

        return False
        
