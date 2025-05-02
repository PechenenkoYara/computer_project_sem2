import random
import numpy as np
from typing import Callable, Tuple, List, Union, TypeVar, Generic, Any, Optional


class Cell:
    """
    A class representing a single cell in the tumor cellular automaton.
    This is a base class that can be extended for different cell types.
    """
    
    def __init__(self, state: int = 0, **kwargs) -> None:
        """
        Initialize a cell with a given state and optional additional properties.
        
        :param state: The numerical state of the cell (e.g., 0=empty, 1=stem, etc.)
        :param kwargs: Additional cell properties that may be needed
        """
        self.state = state
        
        # Store any additional properties passed as kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self) -> str:
        """String representation of the cell."""
        return f"Cell(state={self.state})"
    
    def __eq__(self, other) -> bool:
        """Compare cells based on their state."""
        if isinstance(other, Cell):
            return self.state == other.state
        elif isinstance(other, int):
            return self.state == other
        return False
    
    def copy(self) -> 'Cell':
        """Create a copy of this cell."""
        return Cell(self.state)


# Type variable for cell type
C = TypeVar('C', bound=Cell)


class Grid(Generic[C]):
    """
    A 2D grid for the stochastic cellular automaton tumor model.
    Inspired by Terrain-Generation's Grid (n1n1n1q/Terrain-Generation)  [oai_citation_attribution:2‡GitHub](https://github.com/n1n1n1q/Terrain-Generation/blob/main/src/grid.py).
    """

    def __init__(
        self,
        n: int,
        m: int,
        param_stem: int,
        tumor_creator: Callable[[int], Tuple[np.ndarray, int, Tuple[int, int]]],
        cell_class: type[C] = Cell,
        seed: Optional[str] = None
    ) -> None:
        """
        :param n: number of rows
        :param m: number of cols
        :param param_stem: code for a stem tumor cell (from Valentim model)
        :param tumor_creator: function to create initial tumor
        :param cell_class: the Cell class to use for this grid
        :param seed: optional string seed for reproducibility
        """
        self.n_rows = n
        self.n_cols = m
        self._n = n
        self._m = m
        self.param_stem = param_stem
        self.seed = seed or self._generate_seed()
        self.tumor_creator = tumor_creator
        self.cell_class = cell_class
        self._map: List[List[C]] = []  # will hold our grid of Cell objects
        self.set_up()

    def _generate_seed(self) -> str:
        """Create a random 20-char seed string."""
        chars = "1234567890abcdefghABCDEFGHQWERTYqwerty"
        return ''.join(random.choice(chars) for _ in range(20))

    def set_up(self) -> None:
        """
        Initialize the grid array and place the initial tumor at center.
        """
        random.seed(self.seed)
        np.random.seed()  # let NumPy pick its own if you like, or seed for full reproducibility

        # 1) create an empty grid of Cell objects
        self._map = [[self.cell_class(state=0) for _ in range(self._m)] for _ in range(self._n)]

        # 2) get the small tumor, its size and center
        tumor_array, tumor_size, tumor_center = self.tumor_creator(self.param_stem)

        # 3) embed that tumor matrix into the middle of self._map
        start = ((self._n - tumor_size) // 2, (self._m - tumor_size) // 2)
        i0, j0 = start
        
        # Convert integer values from tumor_array to Cell objects
        for i in range(tumor_size):
            for j in range(tumor_size):
                if tumor_array[i, j] != 0:  # Only place non-zero cells
                    self._map[i0+i][j0+j] = self.cell_class(state=int(tumor_array[i, j]))

    def __getitem__(self, idx: int) -> List[C]:
        """Allow grid[i][j] to return the Cell object."""
        return self._map[idx]
    
    def get_cell(self, i: int, j: int) -> C:
        """Get the Cell at position (i, j)."""
        return self._map[i][j]
    
    def set_cell(self, i: int, j: int, cell: C) -> None:
        """Set the Cell at position (i, j)."""
        self._map[i][j] = cell
    
    def get_cell_state(self, i: int, j: int) -> int:
        """Get the state of the Cell at position (i, j)."""
        return self._map[i][j].state
    
    def set_cell_state(self, i: int, j: int, state: int) -> None:
        """Set the state of the Cell at position (i, j)."""
        self._map[i][j].state = state

    def get_neighbours(self, i: int, j: int) -> List[Tuple[int, int]]:
        """
        Return coordinates of the 8 Moore-neighbours of (i, j),
        clipped at edges.
        """
        neighs = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.n_rows and 0 <= nj < self.n_cols:
                    neighs.append((ni, nj))
        return neighs
    
    def get_neighbour_cells(self, i: int, j: int) -> List[C]:
        """
        Return the actual Cell objects of the 8 Moore-neighbours of (i, j).
        """
        return [self._map[ni][nj] for ni, nj in self.get_neighbours(i, j)]
    
    def to_numpy_array(self) -> np.ndarray:
        """
        Convert the grid of Cell objects to a numpy array of their states.
        Useful for visualization or analysis.
        """
        return np.array([[cell.state for cell in row] for row in self._map], dtype=int)
