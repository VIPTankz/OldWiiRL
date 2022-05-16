import pygame
class Button():
    def __init__(self,x,y,width,height,colour,colourH,border=False,borderC=(0,0,0),
                 text="",font=0,size=30,textColour = (0,0,0),stripe = None,stripeH = None,stripeThickness = None):#Initialise all of the variables
        fonts = pygame.font.get_fonts()
        
        self.x = x #x coordinate of top left corner
        self.y = y #y coordinate of top left corner
        self.width = width #width of the button
        self.height = height #height of the button
        self.colour = colour #colour of button when not hovered over
        self.colourC = colour #current colour of button
        self.colourNH = colour
        self.colourH = colourH #colour of button when hovered over
        self.borderC=borderC #colour of the border of the button, default is black
        self.border=False #Boolean to check if the button has a border
        if border:
            self.border=True
        self.text=text #Text of the button
        self.font=pygame.font.SysFont(fonts[font],size) #font of the button
        self.textColour = textColour
        self.stripeC = stripe
        self.stripe = stripe
        self.stripeH = stripeH
        self.stripeThickness = stripeThickness
        

    def click(self,pos,mouseUp):#method to check if button is pressed
        if self.hovering(pos):#Checks if mouse if over the button
            if mouseUp:#checks if left mouse button was pressed
                return True
            else:
                return False
        else: #Returns false if not over button or not clicked
            return False
        
    def hovering(self,pos):#Method to check if mouse is over the button
        if (pos[0]>self.x and pos[0]<(self.width+self.x) and pos[1]>self.y and pos[1]<(self.height+self.y)):#checks if the mouse is over the button
            self.colourC=self.colourH#changes colour to hover colour
            if self.stripe != None:
                self.stripeC = self.stripeH
            return True
        else:
            self.colourC=self.colour#changes colour back to base colour
            self.stripeC = self.stripe
            return False

    def create(self,screen):#Method to blit the button onto the screen
        if self.border:#Checks if the button has a border
            pygame.draw.rect(screen,self.borderC,(self.x-3,self.y-3,self.width+6,self.height+6),0)#creates a rectangle larger than the button
        pygame.draw.rect(screen,self.colourC,(self.x,self.y,self.width,self.height),0)#draws the button

        if self.stripe != None:
            pygame.draw.rect(screen,self.stripeC,
                             (self.x,self.y + (self.height / 2) - int(self.stripeThickness / 2),self.width,self.stripeThickness),0)

        if self.text != "":#checks if there is text for the button
            text= self.font.render(self.text,1,self.textColour)#creates the text
            screen.blit(text,(self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))#draws the text in the centre of the button

    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getCorners(self):#Returns the coordinates of the four corners of the button
        return self.x,self.y,self.x+self.width,self.y+self.height

    def setCorners(self,co1,co2,co3,co4=(0,0)):#takes four coordinates to set the values of the button
        if co2[0]>co1[0] and co2[1]<co3[1] and co2[1]==co1[1] and co2[0]>co3[0] and co3[0]==co1[0] and co3[1]>co1[1]:#checks if the coordinates create a valid rectangle
            self.x=co1[0]
            self.y=co1[1]#sets the values to create the button with the coordinates given
            self.width=co2[0]-co1[0]
            self.height=co3[1]-co1[1]

    def getWidth(self):#returns the value of the width
        return self.width

    def getHeight(self):#returns the value of the height
        return self.height

    def setWidth(self,width):#Sets the width of the button to the input
        self.width=width

    def setHeight(self,height):#Sets the height of the button to the input
        self.height=height

    def getColour(self):#returns the value of the current colour of the button
        return self.colourC

    def getColourH(self):#returns the value of the current colour of the button
        return self.colourH

    def getColourNH(self):#returns the value of the current colour of the button
        return self.colourNH
    
    def setColour(self,colour):#Sets the base colour of the button to the input
        self.colour=colour

    def setColourH(self,colour):#Sets the hover colour of the button to the input
        self.colourH=colour

    def setBorderColour(self,colour):#Sets the border colour of the button to the input
        self.borderC=colour

    def getText(self):#returns the value of the text for the button
        return self.text

    def setText(self,text):#Sets the text of the button to the input
        self.text=text

    def setFont(self,font,size=-1):#Sets the font of the button to the input
        if size>-1:
            self.font=pygame.font.SysFont(fonts[font],size)
        else:
            self.font=pygame.font.SysFont(fonts[font],26)
