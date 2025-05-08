import random
import numpy as np

class Cell:
    """basic cell, 
    type: 0 - normal cell
          1 - cencer cell
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cct = param_cct  # Cell cycle time
        self.divisions_left = param_reg  # How many divisions are left (0 = stem cell infinite)
        self.time_since_division = 0  # Timer for division
        self.is_alive = True
        
    def update_timer(self, dt):
        self.time_since_division += dt

    def can_divide(self):
        return self.time_since_division >= self.cct

    def reset_timer(self):
        self.time_since_division = 0

    def die(self):
        self.is_alive = False

    def migrate(self, free_neighbors):
        if free_neighbors:
            return random.choice(free_neighbors) 
        return None

    def __str__(self):
        return "Generic Cell"



class NecroticCell(Cell):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.is_alive = False

    def __str__(self):
        return "Necrotic Cell"

class EmptyCell(Cell):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.is_alive = False

    def __str__(self):
        return "Empty"

class RegularTumorCell(Cell):
    """RTC, limited division times, can create only same rtc"""
    def __init__(self, x, y, cct, divisions_left):
        super().__init__(x, y)
        self.cct = cct
        self.divisions_left = divisions_left

    def divide(self, rho=None, param_reg=None, param_stem=None):
        """Завжди створює ще одну RegularTumorCell, p = p_max - 1"""
        if self.divisions_left > 2:
            return RegularTumorCell(self.x, self.y, self.cct, self.divisions_left - 1)
        else:
            return NecroticCell(self.x, self.y)  # клітина помирає

    def __str__(self):
        return "Regular Tumor Cell"


class StemTumorCell(Cell):
    """STC, non limited division potential, can create only RTC"""
    def __init__(self, x, y, cct):
        super().__init__(x, y)
        self.cct = cct
        self.divisions_left = float("inf")

    def divide(self, rho=None, param_reg=None, param_stem=None):
        """Завжди створює RTC, тобто RegularTumorCell"""
        return RegularTumorCell(self.x, self.y, self.cct, param_reg)

    def __str__(self):
        return "Stem Tumor Cell"

class TrueStemCell(Cell):
    """STC, non limited division potential, can create STC or TSC"""

    def __init__(self, x, y, cct):
        super().__init__(x, y)
        self.cct = cct
        self.divisions_left = float("inf")

    def divide(self, rho, param_reg=None, param_stem=None):
        """З ймовірністю rho створює іншу TSC, інакше STC"""
        if random.random() < rho:
            return TrueStemCell(self.x, self.y, self.cct)
        else:
            return StemTumorCell(self.x, self.y, self.cct)

    def __str__(self):
        return "True Stem Cell"




# "Time parameters"
t_max=1000 #Total amount of steps dt
dt = 1/12 # Time step size (fraction of a day) 
# "Model parameters"
param_cct = 24 #cell cycle time (hours)
param_reg = 11 #proliferation potential of regular tumor cell (number of divisions unitl death + 1)
param_stem = param_reg+1 #stem cells have superior proliferation potential (and dont die)
param_potm = 1 #migration potential in cell width  per day
vect_deat,vect_prol,vect_potm,vect_stem = (np.empty(t_max+1) for i in range(4)) #create empty vectors for time-variable chances
vect_deat[:round(0.5*t_max)]=0.01*dt; vect_deat[round(0.5*t_max):]=0.01*dt #chance of death changing w/ time
vect_prol[:] =  (24/param_cct*dt) # Chance of proliferation 
vect_potm[:round(0.4*t_max)] = 10*dt; vect_potm[round(0.4*t_max):] = 10*dt  #Chance of migration changing w/ time
vect_stem[:] = 0.1 #Probability of creating a daughter stem cell