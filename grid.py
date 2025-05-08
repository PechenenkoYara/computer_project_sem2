import random
import numpy as np
from typing import Callable, Tuple


class Grid:
    """
    A 2D grid for the stochastic cellular automaton tumor model.
    Inspired by Terrain-Generation’s Grid (n1n1n1q/Terrain-Generation)  [oai_citation_attribution:2‡GitHub](https://github.com/n1n1n1q/Terrain-Generation/blob/main/src/grid.py).
    """

    def __init__(
        self,
        n: int,
        m: int,
        param_stem: int,
        tumor_creator: Callable[[], Tuple[np.ndarray, int]],
        seed: str | None = None
    ) -> None:
        """
        :param n: number of rows
        :param m: number of cols
        :param param_stem: code for a stem tumor cell (from Valentim model)
        :param seed: optional string seed for reproducibility
        """
        self.n_rows = n
        self.n_cols = m
        self._n = n
        self._m = m
        self.param_stem = param_stem
        self.seed = seed or self._generate_seed()
        self.tumor_creator = tumor_creator
        self._map: np.ndarray[int, np.dtype]  = None  # will hold our grid
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

        # 1) create an empty grid
        self._map = np.zeros((self._n, self._m), dtype=int)

        # 2) get the small tumor, its size and center
        tumor, tumor_size = self.tumor_creator()

        # 3) embed that tumor matrix into the middle of self._map
        start = ((self._n - tumor_size) // 2, (self._m - tumor_size) // 2)
        i0, j0 = start
        self._map[i0:i0+tumor_size, j0:j0+tumor_size] = tumor

    @property
    def map(self):
        """ Public read-only access to the grid map. """
        return self._map

    def __getitem__(self, idx):
        """Allow grid[i][j] to return the cell-state (int)."""
        return self._map[idx]

    def __setitem__(self, idx: Tuple[int, int], value: int) -> None:
        """ Allow grid[i, j] = value syntax to set a cell value.

        Args:
            idx (Tuple[int, int]): The (row, column) indices.
            value (int): The value to set at the specified cell.
        """
        i, j = idx
        self._map[i, j] = value

    def get_neighbours(self, i: int, j: int) -> list[tuple[int, int]]:
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
