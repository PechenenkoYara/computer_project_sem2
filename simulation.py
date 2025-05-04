"""
Fixed tumor simulation module with improved constants usage for cell types
"""

import random
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from grid import Grid

EMPTY = 0
TUMOR = 1
NECROTIC = 2
STEM = 3

@njit
def simulate_step(cell_type_matrix: np.ndarray, division_potential: np.ndarray,
                  cell_timer: np.ndarray, active_cells_arr: np.ndarray,
                  time_step: float, cell_cycle_time: float,
                  regular_cell_division_potential: int, stem_cell_potential: int,
                  death_prob: float, prolif_prob: float, migrate_prob: float, stem_prob: float,
                  rows: int, cols: int) -> Tuple[int, np.ndarray, int]:
    """
    Simulates a single time step of the tumor growth simulation.

    Args:
        cell_type_matrix (np.ndarray): Matrix containing cell type identifiers.
        division_potential (np.ndarray): Matrix containing remaining division potential for each cell.
        cell_timer (np.ndarray): Matrix containing timer values for cell cycle progression.
        active_cells_arr (np.ndarray): Array of (i,j) coordinates of active cells.
        time_step (float): Size of each time step in days.
        cell_cycle_time (float): Time required (in hours) for a cell to divide once.
        regular_cell_division_potential (int): Maximum divisions for regular tumor cells.
        stem_cell_potential (int): Maximum divisions for stem cells.
        death_prob (float): Probability of cell death per time step.
        prolif_prob (float): Probability of cell proliferation per time step.
        migrate_prob (float): Probability of cell migration per time step.
        stem_prob (float): Probability of stem cell creation during division.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        Tuple[int, np.ndarray, int]: A tuple containing:
            - Count of new necrotic cells created in this step
            - Array of coordinates for new active cells
            - Count of new stem cells created in this step
    """
    necrotic_count = 0
    new_active_cells = []
    stem_created = 0

    if active_cells_arr.shape[0] == 0:
        return necrotic_count, np.array(new_active_cells, dtype=np.int32), stem_created

    for idx in range(active_cells_arr.shape[0]):
        if idx >= active_cells_arr.shape[0]:
            continue

        i, j = active_cells_arr[idx]

        if not (0 <= i < rows and 0 <= j < cols):
            continue

        if cell_type_matrix[i, j] in (EMPTY, NECROTIC):
            continue

        cell_timer[i, j] += time_step

        # Random events
        r_death = np.random.rand()
        r_prolif = np.random.rand()
        r_migrate = np.random.rand()

        # Death - only for non-stem cells
        if cell_type_matrix[i, j] != STEM and r_death < death_prob:
            cell_type_matrix[i, j] = NECROTIC
            division_potential[i, j] = 0
            cell_timer[i, j] = 0.0
            necrotic_count += 1
            continue

        # Proliferation
        if cell_timer[i, j] >= cell_cycle_time and r_prolif < prolif_prob:
            neighbors = get_empty_neighbors(cell_type_matrix, i, j, rows, cols)
            if neighbors.shape[0] > 0:
                neighbor_idx = np.random.randint(0, neighbors.shape[0])
                if neighbor_idx >= neighbors.shape[0]:
                    continue

                ni, nj = neighbors[neighbor_idx]

                if 0 <= ni < rows and 0 <= nj < cols:
                    if cell_type_matrix[i, j] == STEM:
                        r_random = np.random.rand()
                        if r_random < stem_prob:
                            cell_type_matrix[ni, nj] = STEM
                            division_potential[ni, nj] = stem_cell_potential
                            stem_created += 1
                        else:
                            cell_type_matrix[ni, nj] = TUMOR
                            division_potential[ni, nj] = regular_cell_division_potential
                    else:
                        cell_type_matrix[ni, nj] = TUMOR
                        division_potential[ni, nj] = regular_cell_division_potential

                    cell_timer[ni, nj] = 0.0
                    new_active_cells.append((ni, nj))

                    cell_timer[i, j] = 0.0

                    if cell_type_matrix[i, j] != STEM:
                        division_potential[i, j] -= 1
                        if division_potential[i, j] <= 0:
                            cell_type_matrix[i, j] = NECROTIC
                            division_potential[i, j] = 0
                            cell_timer[i, j] = 0.0
                            necrotic_count += 1
                            continue

        # Migration
        elif r_migrate < migrate_prob:
            neighbors = get_empty_neighbors(cell_type_matrix, i, j, rows, cols)
            if neighbors.shape[0] > 0:
                neighbor_idx = np.random.randint(0, neighbors.shape[0])
                if neighbor_idx >= neighbors.shape[0]:
                    continue

                ni, nj = neighbors[neighbor_idx]

                if 0 <= ni < rows and 0 <= nj < cols:
                    cell_type = cell_type_matrix[i, j]
                    div = division_potential[i, j]
                    timer = cell_timer[i, j]

                    cell_type_matrix[i, j] = EMPTY
                    division_potential[i, j] = 0
                    cell_timer[i, j] = 0.0

                    cell_type_matrix[ni, nj] = cell_type
                    division_potential[ni, nj] = div
                    cell_timer[ni, nj] = timer

                    new_active_cells.append((ni, nj))

    if not new_active_cells:
        return necrotic_count, np.empty((0, 2), dtype=np.int32), stem_created
    return necrotic_count, np.array(new_active_cells, dtype=np.int32), stem_created

@njit
def get_empty_neighbors(cell_type_matrix: np.ndarray, i: int, j: int,
                       rows: int, cols: int) -> np.ndarray:
    """
    Identify all adjacent empty cells around a given cell.

    Args:
        cell_type_matrix (np.ndarray): Matrix containing cell type identifiers.
        i (int): Row index of the current cell.
        j (int): Column index of the current cell.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        np.ndarray: An array of (row, column) coordinates representing
        neighboring grid positions that are currently empty.
    """
    neighbors = np.empty((8, 2), dtype=np.int32)
    count = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ni = i + dx
            nj = j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                if cell_type_matrix[ni, nj] == EMPTY:
                    if count < 8:
                        neighbors[count, 0] = ni
                        neighbors[count, 1] = nj
                        count += 1
    return neighbors[:count]

class TumorSimulation:
    """
    Simulation class to handle the time evolution of the tumor model.
    Enhanced version with optimized stem cell creation and tracking.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        maximum_time_steps: int = 1000,
        time_step: float = 1/12,
        cell_cycle_time: float = 24,
        regular_cell_division_potential: int = 11,
        stem_cell_potential: int = 12,
        migration_potential: int = 1,
        tumor_size: int = 10,
        seed: Optional[int] = None,
        run_statistics: bool = False,
        statistics_step: int = 10
    ):
        """
        Initialize the tumor simulation environment.

        Args:
            grid_size (Tuple[int, int]): Dimensions of the 2D grid (rows, columns).
            maximum_time_steps (int): Number of total discrete time steps in the simulation.
            time_step (float): Size of each time step in days (e.g., 1/12 = 2 hours).
            cell_cycle_time (float): Time required (in hours) for a cell to divide once.
            regular_cell_division_potential (int): Maximum number of divisions a regular tumor cell can perform.
            stem_cell_potential (int): Maximum number of divisions a stem cell can perform.
            migration_potential (int): Migration activity level of a cell per day.
            tumor_size (int): Size of the square tumor matrix.
            seed (Optional[int]): Seed for reproducibility of random events.
            run_statistics (bool): If True, statistics will be saved. 
                If False, statistics will not be saved, which improves performance.
            statistics_step (int): Statistics recording frequency.
        """
        self.n_rows, self.n_cols = grid_size
        self.maximum_time_steps = maximum_time_steps
        self.time_step = time_step
        self.cell_cycle_time = cell_cycle_time
        self.regular_cell_division_potential = regular_cell_division_potential
        self.stem_cell_potential = stem_cell_potential
        self.migration_potential = migration_potential
        self.tumor_size = tumor_size
        self.run_statistics = run_statistics
        self.statistics_step = statistics_step

        self.grid = Grid(
            n=self.n_rows,
            m=self.n_cols,
            param_stem=self.stem_cell_potential,
            tumor_creator=lambda stem: self.__create_initial_tumor(stem, self.tumor_size),
            seed=seed
        )

        self.cell_type_matrix = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        self.division_potential = np.zeros((self.n_rows, self.n_cols), dtype=np.int16)
        self.cell_timer = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
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
        # Represents the probability of creating stem cells during division
        # Increased from 0.1 to 0.3 in this optimized version
        self.vect_stem = np.full(maximum_time_steps + 1, 0.3)

        self.stats = {
            'time': np.zeros(maximum_time_steps + 1),
            'total_cells': np.zeros(maximum_time_steps + 1, dtype=int),
            'tumor_cells': np.zeros(maximum_time_steps + 1, dtype=int),
            'stem_cells': np.zeros(maximum_time_steps + 1, dtype=int),
            'necrotic_cells': np.zeros(maximum_time_steps + 1, dtype=int),
            'new_stem_cells': np.zeros(maximum_time_steps + 1, dtype=int)
        }

        self.stats_index = 0
        self.necrotic_count = 0
        self.stem_created_count = 0
        self.current_step = 0
        self.current_time = 0.0

    def __create_initial_tumor(self, stem_cell_potential: int,\
                    tumor_size: int) -> Tuple[np.ndarray, int, Tuple[int, int]]:
        """
        Creates initial tumor configuration.

        Args:
            stem_cell_potential (int): Proliferation potential of the central stem cell.
            tumor_size (int): Size of the square tumor matrix.

        Returns:
            Tuple[np.ndarray, int, Tuple[int, int]]: A tuple containing:
                - tumor (np.ndarray): Grid with initial tumor cells.
                - tumor_size (int): The size of the tumor matrix.
                - center_coordinates (Tuple[int, int]): Coordinates of the central cell.
        """
        tumor = np.zeros((tumor_size, tumor_size), dtype=int)
        center = tumor_size // 2
        tumor[center, center] = stem_cell_potential

        for i in range(tumor_size):
            for j in range(tumor_size):
                if (i, j) != (center, center) and random.random() < 0.3:
                    tumor[i, j] = 1

        return tumor, tumor_size, (center, center)

    def __initialize_cells_from_grid(self) -> None:
        """
        Creates cell representations based on the initial grid configuration.
        Converts grid values to appropriate cell types in the simulation matrices.
        """
        initial_stem_count = 0

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                val = self.grid[i][j]
                if val == 0:
                    continue
                elif val == 1:
                    self.cell_type_matrix[i, j] = TUMOR
                    self.division_potential[i, j] = self.regular_cell_division_potential
                    self.active_cells.add((i, j))
                elif val == self.stem_cell_potential:
                    self.cell_type_matrix[i, j] = STEM
                    self.division_potential[i, j] = self.stem_cell_potential
                    self.active_cells.add((i, j))
                    initial_stem_count += 1

        print(f"Initialized with {initial_stem_count} stem cells")

    def update_grid_cell(self, i: int, j: int, cell_type: int,
                         divisions_left: int = 0, timer: float = 0.0) -> None:
        """
        Update the simulation state at a specific grid location.

        Args:
            i (int): Row index of the cell to update.
            j (int): Column index of the cell to update.
            cell_type (int): The type of cell to place at position (i, j).
            divisions_left (int, optional): Number of divisions remaining for the cell.
            timer (float, optional): Current cell cycle timer value.
        """
        if not (0 <= i < self.n_rows and 0 <= j < self.n_cols):
            return

        self.cell_type_matrix[i, j] = cell_type
        self.division_potential[i, j] = divisions_left
        self.cell_timer[i, j] = timer
        pos = (i, j)
        if cell_type in (EMPTY, NECROTIC):
            self.active_cells.discard(pos)
        else:
            self.active_cells.add(pos)

    def __run_step(self) -> bool:
        """
        Executes a single time step of the tumor simulation.

        Simulates the behavior of all cells in the grid for one discrete time unit.
        It applies biological processes such as death, proliferation (division), and migration
        according to time-dependent probabilities.

        Returns:
            bool: True if simulation should continue, False otherwise.
        """
        if self.active_cells:
            active_array = np.array(list(self.active_cells), dtype=np.int32)
        else:
            return False

        current_step = min(self.current_step, len(self.vect_deat) - 1)

        necrotic_this_step, new_cells, stem_created = simulate_step(
            self.cell_type_matrix,
            self.division_potential,
            self.cell_timer,
            active_array,
            self.time_step,
            self.cell_cycle_time,
            self.regular_cell_division_potential,
            self.stem_cell_potential,
            self.vect_deat[current_step],
            self.vect_prol[current_step],
            self.vect_potm[current_step],
            self.vect_stem[current_step],
            self.n_rows,
            self.n_cols
        )

        self.necrotic_count += necrotic_this_step
        self.stem_created_count += stem_created

        new_active_cells = set()
        for i, j in self.active_cells:
            if 0 <= i < self.n_rows and 0 <= j < self.n_cols:
                if self.cell_type_matrix[i, j] in (TUMOR, STEM):
                    new_active_cells.add((i, j))

        for i, j in new_cells:
            if 0 <= i < self.n_rows and 0 <= j < self.n_cols:
                if self.cell_type_matrix[i, j] in (TUMOR, STEM):
                    new_active_cells.add((i, j))

        self.active_cells = new_active_cells

        if not self.active_cells:
            return False

        if self.run_statistics and self.current_step % self.statistics_step == 0:
            self.__record_statistics()

        self.current_step += 1
        self.current_time += self.time_step

        if self.current_step % 100 == 0:
            stem_count = sum(1 for i, j in self.active_cells
                          if 0 <= i < self.n_rows and 0 <= j < self.n_cols
                          and self.cell_type_matrix[i, j] == STEM)

            print(f"{self.current_step}/{self.maximum_time_steps} - "
                  f"Active cells: {len(self.active_cells)}, Stem cells: {stem_count}")

        return self.current_step <= self.maximum_time_steps and len(self.active_cells) > 0

    def run_simulation(self) -> None:
        """
        Runs the tumor simulation through all configured time steps.

        Iteratively advances the simulation one step at a time until the
        maximum number of time steps is reached or no active cells remain.
        """
        while self.__run_step():
            pass

        if self.run_statistics:
            self.plot_statistics()
            plt.savefig('tumor_stats.png')

    def __record_statistics(self) -> None:
        """
        Records the current state of the simulation, including the count of 
        total cells, tumor cells, stem cells, and necrotic cells.
        Also tracks newly created stem cells.
        """
        total, tumor, stem = 0, 0, 0
        for i, j in self.active_cells:
            if not (0 <= i < self.n_rows and 0 <= j < self.n_cols):
                continue

            ctype = self.cell_type_matrix[i, j]
            total += 1
            if ctype == STEM:
                stem += 1
                tumor += 1
            elif ctype == TUMOR:
                tumor += 1

        if self.stats_index >= len(self.stats['time']):
            return

        self.stats['time'][self.stats_index] = self.current_time
        self.stats['total_cells'][self.stats_index] = total + self.necrotic_count
        self.stats['tumor_cells'][self.stats_index] = tumor
        self.stats['stem_cells'][self.stats_index] = stem
        self.stats['necrotic_cells'][self.stats_index] = self.necrotic_count
        self.stats['new_stem_cells'][self.stats_index] = self.stem_created_count
        self.stats_index += 1

    def plot_statistics(self) -> plt.Figure:
        """
        Plots the evolution of key statistics
        (total cells, tumor cells, stem cells, necrotic cells) over time.
        
        Returns:
            plt.Figure: The generated matplotlib figure.
        """
        plt.figure(figsize=(12, 8))
        valid_idx = min(self.stats_index, len(self.stats['time']))
        t = self.stats['time'][:valid_idx]

        plt.plot(t, self.stats['total_cells'][:valid_idx], 'k-', label='Total Cells', linewidth=2)
        plt.plot(t, self.stats['tumor_cells'][:valid_idx], 'r-', label='Tumor Cells', linewidth=2)
        plt.plot(t, self.stats['stem_cells'][:valid_idx], 'g-', label='Stem Cells', linewidth=2)
        plt.plot(t, self.stats['necrotic_cells'][:valid_idx], 'b-', label='Necrotic Cells', linewidth=2)
        plt.plot(t, self.stats['new_stem_cells'][:valid_idx], 'm--', label='New Stem Cells Created', linewidth=2)

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
        maximum_time_steps=5000,
        time_step=1/12,
        cell_cycle_time=24,
        regular_cell_division_potential=11,
        stem_cell_potential=12,
        migration_potential=1,
        tumor_size=10,
        run_statistics=True,
        statistics_step=10,
    )

    sim.run_simulation()
