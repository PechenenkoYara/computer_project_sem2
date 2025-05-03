"""
Tumor simulation module
"""

import random

from typing import Tuple, Optional

from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from cell import Cell, StemCell, NecroticCell, TumorCell, EmptyCell
from grid import Grid

SHARED_EMPTY_CELL = EmptyCell(-1, -1)
class TumorSimulation:
    """
    Simulation class to handle the time evolution of the tumor model.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        maximum_time_steps: int = 1000,
        time_step: float = 1/12,
        cell_cycle_time: int = 24,
        regular_cell_division_potentional: int = 11,
        stem_cell_potentional: int = 12,
        migration_potentional: int = 1,
        tumor_size: int = 10,
        seed: Optional[str] = None,
        run_statistics: bool = False,
        statistics_step: int = 10
    ):
        """
        Initialize the tumor simulation environment.

        Args:
            grid_size (Tuple[int, int]): Dimensions of the 2D grid (rows, columns).
            maximum_time_steps (int): Number of total discrete time steps in the simulation.
            time_step (float): Size of each time step in days (e.g., 1/12 = 2 hours).
            cell_cycle_time (int): Time required (in hours) for a cell to divide once.
            regular_cell_division_potentional (int): Maximum number of divisions a regular tumor cell can perform.
            stem_cell_potentional (int): Maximum number of divisions a stem cell can perform.
            migration_potentional (int): Migration activity level of a cell per day.
            tumor_size (int): Size of the square tumor matrix.
            seed (Optional[str]): Seed for reproducibility of random events.
            run_statistics (bool): If True, statistics will be saved. 
            If False, statistics will not be saved, which improves performance.
            statistics_step (int): Statistics recording frequency.
        """

        self.n_rows, self.n_cols = grid_size
        self.maximum_time_steps = maximum_time_steps
        self.time_step = time_step
        self.cell_cycle_time = cell_cycle_time
        self.regular_cell_division_potentional = regular_cell_division_potentional
        self.stem_cell_potentional = stem_cell_potentional
        self.migration_potentional = migration_potentional
        self.tumor_size = tumor_size
        self.run_statistics = run_statistics
        self.statistics_step = statistics_step

        self.grid = Grid(
            n=self.n_rows,
            m=self.n_cols,
            param_stem=self.stem_cell_potentional,
            tumor_creator=lambda stem: self.__create_initial_tumor(stem, self.tumor_size),
            seed=seed
        )

        self.cell_matrix = np.full((self.n_rows, self.n_cols), SHARED_EMPTY_CELL, dtype=object)
        self.active_cells = set()
        self.__initialize_cells_from_grid()

        # --- Time-variable parameter arrays ---

        # Death rate over time (per time step).
        # Represents the probability or rate at which cells die.
        self.vect_deat = np.full(maximum_time_steps + 1, 0.01 * time_step)

        # Proliferation rate over time.
        # Scales with how often cells divide, normalized to simulation time steps.
        # Formula: (24 hours / cell_cycle_time) × time_step.
        self.vect_prol = np.full(maximum_time_steps + 1, (24 / cell_cycle_time) * time_step)

        # Migration potential over time. Represents how likely or far a cell can migrate.
        # Currently fixed to 10 × time_step throughout the simulation.
        self.vect_potm = np.full(maximum_time_steps + 1, 10 * time_step)

        # Stem cell behavior parameter over time.
        # Could represent differentiation rate, plasticity, or activation rate
        self.vect_stem = np.full(maximum_time_steps + 1, 0.1)

        self.stats = {
            'time': np.zeros(maximum_time_steps + 1),
            'total_cells': np.zeros(maximum_time_steps + 1, dtype=int),
            'tumor_cells': np.zeros(maximum_time_steps + 1, dtype=int),
            'stem_cells': np.zeros(maximum_time_steps + 1, dtype=int),
            'necrotic_cells': np.zeros(maximum_time_steps + 1, dtype=int)
        }

        self.stats_index = 0
        self.necrotic_count = 0
        self.current_step = 0
        self.current_time = 0.0

    def __create_initial_tumor(self, stem_cell_potentional: int, tumor_size: int):
        """ Creates initial tumor configuration.

        Args:
            stem_cell_potentional (int): Proliferation potential of the central stem cell.
            tumor_size (int): Size of the square tumor matrix.

        Returns:
            tuple:
                tumor_matrix (np.ndarray): Grid with initial tumor cells.
                size (int): The size of the tumor matrix.
                center_coordinates (Tuple[int, int]): Coordinates of the central cell.
        """

        tumor = np.zeros((tumor_size, tumor_size), dtype=int)
        center = tumor_size // 2
        tumor[center, center] = stem_cell_potentional

        for i in range(tumor_size):
            for j in range(tumor_size):
                if (i, j) != (center, center) and random.random() < 0.3:
                    tumor[i, j] = 1

        return tumor, tumor_size, (center, center)

    def __initialize_cells_from_grid(self):
        """ Creates cell objects based on the initial grid configuration.
        """

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                cell_type = self.grid[i][j]
                pos = (i, j)

                if cell_type == 0:
                    cell = self.cell_matrix[i, j] = SHARED_EMPTY_CELL
                elif cell_type == 1:
                    cell = TumorCell(i, j, \
            self.cell_cycle_time, self.regular_cell_division_potentional)
                    cell = self.cell_matrix[i, j] = cell
                    self.active_cells.add(pos)
                elif cell_type == self.stem_cell_potentional:
                    cell = StemCell(i, j, self.cell_cycle_time)
                    cell = self.cell_matrix[i, j] = cell
                    self.active_cells.add(pos)

    def get_empty_neighbors(self, i: int, j: int):
        """ Identify all adjacent empty cells around a given cell.

        Args:
            i (int): Row index of the current cell.
            j (int): Column index of the current cell.

        Returns:
            List[Tuple[int, int]]: A list of (row, column) tuples representing
            neighboring grid positions that are currently empty (i.e., contain an EmptyCell object).
        """

        return [(ni, nj) for ni, nj in self.grid.get_neighbours(i, j)
            if self.cell_matrix[ni, nj] == SHARED_EMPTY_CELL]

    def update_grid_cell(self, i: int, j: int, cell_obj: Cell):
        """
        Update the simulation state at a specific grid location.

        Args:
            i (int): Row index of the cell to update.
            j (int): Column index of the cell to update.
            cell_obj (Cell): The new cell object to place at position (i, j).
        """

        pos = (i, j)
        if isinstance(cell_obj, EmptyCell):
            self.grid[i, j] = 0
            self.active_cells.discard(pos)
        elif isinstance(cell_obj, NecroticCell):
            self.grid[i, j] = 2
            self.active_cells.discard(pos)
        elif isinstance(cell_obj, StemCell):
            self.grid[i, j] = self.stem_cell_potentional
            self.active_cells.add(pos)
        elif isinstance(cell_obj, TumorCell):
            self.grid[i, j] = 1
            self.active_cells.add(pos)

        self.cell_matrix[i, j] = cell_obj

    def __run_step(self):
        """ Executes a single time step of the tumor simulation.

        Simulates the behavior of all cells in the grid for one discrete time unit.
        It applies biological processes such as death, proliferation (division), and migration
        according to time-dependent probabilities.
        """

        death_prob = self.vect_deat[self.current_step]
        prolif_prob = self.vect_prol[self.current_step]
        migrate_prob = self.vect_potm[self.current_step]
        stem_prob = self.vect_stem[self.current_step]

        death_random_values = {pos: random.random() for pos in self.active_cells}
        prolif_random_values = {pos: random.random() for pos in self.active_cells}
        migrate_random_values = {pos: random.random() for pos in self.active_cells}

        for pos in list(self.active_cells):
            i, j = pos
            cell = self.cell_matrix[i, j]
            if cell.is_alive:
                cell.update_timer(self.time_step)

        for pos in list(self.active_cells):
            i, j = pos
            cell = self.cell_matrix[i, j]

            if not cell.is_alive:
                continue

            # Death
            if death_random_values[pos] < death_prob:
                self.update_grid_cell(i, j, NecroticCell(i, j))
                self.necrotic_count += 1
                continue

            # Division
            if cell.can_divide() and prolif_random_values[pos] < prolif_prob:
                empty_neighbors = self.get_empty_neighbors(i, j)
                if empty_neighbors:
                    ni, nj = random.choice(empty_neighbors)
                    cell.reset_timer()

                    if isinstance(cell, TumorCell) and not isinstance(cell, StemCell):
                        cell.divisions_left -= 1
                        if cell.divisions_left <= 0:
                            self.update_grid_cell(i, j, NecroticCell(i, j))
                            continue

                    daughter = cell.divide(
                        stem_prob,
                        self.regular_cell_division_potentional,
                        self.stem_cell_potentional
                    )
                    self.update_grid_cell(ni, nj, daughter)

            # Migration
            if migrate_random_values[pos] < migrate_prob:
                empty_neighbors = self.get_empty_neighbors(i, j)
                new_pos = cell.migrate(empty_neighbors)
                if new_pos:
                    ni, nj = new_pos
                    self.update_grid_cell(i, j, SHARED_EMPTY_CELL)
                    cell.x, cell.y = ni, nj
                    self.update_grid_cell(ni, nj, cell)

        if self.run_statistics and self.current_step % self.statistics_step == 0:
            self.__record_statistics()

        self.current_step += 1
        self.current_time += self.time_step

        if self.current_step % 100 == 0:
            print(f"{self.current_step}/{self.maximum_time_steps}")

        return self.current_step <= self.maximum_time_steps

    def run_simulation(self):
        """ Runs the tumor simulation through all configured time steps.

        Iteratively advances the simulation one step at a time until the
        maximum number of time steps is reached.
        """

        while self.__run_step():
            pass

        if self.run_statistics:
            self.plot_statistics()
            plt.savefig('tumor_stats.png')

    def __record_statistics(self):
        """ Records the current state of the simulation, including the count of 
        total cells, tumor cells, stem cells, and necrotic cells.
        """

        total, tumor, stem = 0, 0, 0
        for pos in self.active_cells:
            i, j = pos
            cell = self.cell_matrix[i, j]
            total += 1
            if isinstance(cell, StemCell):
                stem += 1
                tumor += 1
            elif isinstance(cell, TumorCell):
                tumor += 1

        self.stats['time'][self.stats_index] = self.current_time
        self.stats['total_cells'][self.stats_index] = total + self.necrotic_count
        self.stats['tumor_cells'][self.stats_index] = tumor
        self.stats['stem_cells'][self.stats_index] = stem
        self.stats['necrotic_cells'][self.stats_index] = self.necrotic_count

        self.stats_index += 1

    def plot_statistics(self):
        """ Plots the evolution of key statistics
        (total cells, tumor cells, stem cells, necrotic cells) over time.
        """

        plt.figure(figsize=(12, 8))

        t = self.stats['time'][:self.stats_index]

        plt.plot(t, self.stats['total_cells'][:self.stats_index],\
                  'k-', label='Total Cells', linewidth=2)
        plt.plot(t, self.stats['tumor_cells'][:self.stats_index],\
                  'r-', label='Tumor Cells', linewidth=2)
        plt.plot(t, self.stats['stem_cells'][:self.stats_index],\
                  'g-', label='Stem Cells', linewidth=2)
        plt.plot(t, self.stats['necrotic_cells'][:self.stats_index],\
                  'b-', label='Necrotic Cells', linewidth=2)

        plt.xlabel('Time (days)', fontsize=14)
        plt.ylabel('Cell Count', fontsize=14)
        plt.title('Tumor Growth Simulation Statistics', fontsize=16)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return plt.gcf()

if __name__ == "__main__":
    sim = TumorSimulation(
        grid_size=(1000, 1000),
        maximum_time_steps=1000,
        time_step=1/12,
        cell_cycle_time=24,
        regular_cell_division_potentional=11,
        stem_cell_potentional=12,
        migration_potentional=1,
        tumor_size=10,
        run_statistics=True,
        statistics_step=10
    )

    sim.run_simulation()
