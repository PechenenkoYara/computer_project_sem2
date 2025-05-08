"""
Cell Module
"""

import random
import numpy as np

class Cell:
    """
    Base class for all cell types in the simulation.

    Attributes:
        x (int): X-coordinate of the cell.
        y (int): Y-coordinate of the cell.
        cct (float): Cell cycle time (how long until the cell can divide).
        divisions_left (int or float): Number of divisions left; infinite for stem cells.
        time_since_division (float): Time passed since last division.
        is_alive (bool): Flag indicating if the cell is alive.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cct = 24
        self.divisions_left = 11
        self.time_since_division = 0
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
    """
    Represents a dead (necrotic) cell.

    Always non-dividing and not alive.
    """

    def __init__(self, x, y):
        super().__init__(x, y)
        self.is_alive = False

    def __str__(self):
        return "Necrotic Cell"

class EmptyCell(Cell):
    """
    Represents an empty location in the grid.

    Used as placeholder for unoccupied positions.
    """

    def __init__(self, x, y):
        super().__init__(x, y)
        self.is_alive = False

    def __str__(self):
        return "Empty"

class RegularTumorCell(Cell):
    """
    Tumor cell with limited division capability.

    Attributes:
        cct (float): Cell cycle time.
        divisions_left (int): How many times the cell can divide.

    Methods:
        divide(): Returns a new RegularTumorCell or NecroticCell if no divisions remain.
    """

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
    """
    Tumor stem-like cell with unlimited division potential.

    Attributes:
        cct (float): Cell cycle time.

    Methods:
        divide(): Always returns a new RegularTumorCell.
    """

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
    """
    True stem cell that can produce either another TrueStemCell or a StemTumorCell.

    Attributes:
        cct (float): Cell cycle time.

    Methods:
        divide(rho): With probability rho creates another TrueStemCell, else StemTumorCell.
    """

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
