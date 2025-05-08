"""
Tumor Simulation Module
"""

import random
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from cell import (Cell, StemTumorCell, TrueStemCell,
                NecroticCell, RegularTumorCell, EmptyCell, ImmuneCell)

from grid import Grid

SHARED_EMPTY_CELL = EmptyCell(-1, -1)

class TumorSimulation:
    """
    Simulation class to handle the time evolution of the tumor model with immune system interaction.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        max_simulation_days: int = 1000,
        time_step: float = 1/12,
        cell_cycle_time: int = 24,
        regular_cell_division_potentional: int = 11,
        stem_cell_potentional: int = 12,
        migration_potentional: int = 1,
        tumor_size: int = 10,
        TSC_prob: float = 0.1,
        STC_prob: float = 0.2,
        RST_prob: float = 0.5,

        # Immune system parameters
        immune_activation_day: int = 5,
        immune_cell_count: int = 10,
        immune_cell_lifespan: int = 72,
        immune_cell_division_potential: int = 3,
        immune_kill_probability: float = 0.3,
        immune_migration_factor: float = 2.0,
        enable_immune_system: bool = False,

        # Other settings
        seed: Optional[str] = None,
        run_statistics: bool = False,
        statistics_step: int = 10,
        plot_interval: int = 100
    ):
        """
        Initialize the tumor simulation environment with immune system capabilities.

        Args:
            grid_size (Tuple[int, int]): Dimensions of the 2D grid (rows, columns).
            maximum_time_steps (int): Number of total discrete time steps in the simulation.
            time_step (float): Size of each time step in days (e.g., 1/12 = 2 hours).
            cell_cycle_time (int): Time required (in hours) for a cell to divide once.
            regular_cell_division_potentional (int): Maximum number of divisions a regular tumor cell can perform.
            stem_cell_potentional (int): Maximum number of divisions a stem cell can perform.
            migration_potentional (int): Migration activity level of a cell per day.
            tumor_size (int): Size of the square tumor matrix.
            immune_activation_day (int): Day when immune response starts.
            immune_cell_count (int): Number of immune cells to introduce.
            immune_cell_lifespan (int): Lifespan of immune cells in hours.
            immune_cell_division_potential (int): Number of divisions immune cells can perform.
            immune_kill_probability (float): Base probability for immune cells to kill tumor cells.
            immune_migration_factor (float): How much more active immune cells are at migration compared to tumor cells.
            enable_immune_system (bool): Whether to enable immune system in simulation.
            seed (Optional[str]): Seed for reproducibility of random events.
            run_statistics (bool): If True, statistics will be saved.
            statistics_step (int): Statistics recording frequency.
            plot_interval (int): Interval of days between visualization plots.
        """

        self.n_rows, self.n_cols = grid_size
        self.maximum_time_steps = int(max_simulation_days / time_step)
        self.time_step = time_step
        self.cell_cycle_time = cell_cycle_time
        self.regular_cell_division_potentional = regular_cell_division_potentional + 2
        self.stem_cell_potentional = stem_cell_potentional
        self.migration_potentional = migration_potentional
        self.tumor_size = tumor_size
        self.TSC_prob = TSC_prob
        self.STC_prob = STC_prob
        self.RST_prob = RST_prob

        # Immune system parameters
        self.immune_activation_day = immune_activation_day
        self.immune_activation_step = int(immune_activation_day / time_step)
        self.immune_cell_count = immune_cell_count
        self.immune_cell_lifespan = immune_cell_lifespan
        self.immune_cell_division_potential = immune_cell_division_potential
        self.immune_kill_probability = immune_kill_probability
        self.immune_migration_factor = immune_migration_factor
        self.enable_immune_system = enable_immune_system

        self.run_statistics = run_statistics
        self.statistics_step = statistics_step
        self.plot_interval = plot_interval

        self.grid = Grid(
            n=self.n_rows,
            m=self.n_cols,
            param_stem=self.stem_cell_potentional,
            tumor_creator=lambda: self.__create_initial_tumor(\
            self.regular_cell_division_potentional, self.tumor_size,\
            self.TSC_prob, self.STC_prob, self.RST_prob),

            seed=seed
        )

        self.cell_matrix = np.full((self.n_rows, self.n_cols), SHARED_EMPTY_CELL, dtype=object)
        self.active_cells = set()
        self.immune_cells = set()  # Track immune cell positions separately
        self.__initialize_cells_from_grid()

        # --- Time-variable parameter arrays ---

        # Death rate over time (per time step)
        self.vect_deat = np.full(self.maximum_time_steps + 1, 0.01 * time_step)

        # Proliferation rate over time
        self.vect_prol = np.full(self.maximum_time_steps + 1, (24 / cell_cycle_time) * time_step)

        # Migration potential over time
        self.vect_potm = np.full(self.maximum_time_steps + 1, 10 * time_step)

        # Stem cell behavior parameter over time
        self.vect_stem = np.full(self.maximum_time_steps + 1, 0.1)

        # Immune activation over time (starts at 0, increases after activation day)
        self.vect_immune = np.zeros(self.maximum_time_steps + 1)
        if self.enable_immune_system:
            self.vect_immune[self.immune_activation_step:] = 1.0

        self.stats = {
            'time': np.zeros(self.maximum_time_steps + 1),
            'total_cells': np.zeros(self.maximum_time_steps + 1, dtype=int),
            'true_stems': np.zeros(self.maximum_time_steps + 1, dtype=int),
            'tumor_cells': np.zeros(self.maximum_time_steps + 1, dtype=int),
            'stem_cells': np.zeros(self.maximum_time_steps + 1, dtype=int),
            'necrotic_cells': np.zeros(self.maximum_time_steps + 1, dtype=int),
            'immune_cells': np.zeros(self.maximum_time_steps + 1, dtype=int),
            'killed_tumor_cells': np.zeros(self.maximum_time_steps + 1, dtype=int)
        }

        self.stats_index = 0
        self.necrotic_count = 0
        self.killed_by_immune_count = 0
        self.current_step = 0
        self.current_time = 0.0
        self.last_plot_day = 0

    def __create_initial_tumor(self,
        stem_cell_potentional: int,
        tumor_size: int,
        TSC_prob: float,
        STC_prob: float,
        RTC_prob: float):
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

        for i in range(tumor_size):
            for j in range(tumor_size):
                prob = random.random()
                if prob < TSC_prob:
                    tumor[i, j] = 1
                    continue
                if prob < STC_prob:
                    tumor[i, j] = 2
                    continue
                if prob < RTC_prob:
                    tumor[i, j] = stem_cell_potentional

        return tumor, tumor_size

    def __initialize_cells_from_grid(self):
        """ Creates cell objects based on the initial grid configuration."""

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                cell_type = self.grid[i][j]
                pos = (i, j)

                if cell_type == 0:
                    self.cell_matrix[i, j] = SHARED_EMPTY_CELL
                elif cell_type == 1:
                    cell = TrueStemCell(i, j, self.cell_cycle_time)
                    self.cell_matrix[i, j] = cell
                    self.active_cells.add(pos)
                elif cell_type == 2:
                    cell = StemTumorCell(i, j, self.cell_cycle_time)
                    self.cell_matrix[i, j] = cell
                    self.active_cells.add(pos)
                elif cell_type == self.regular_cell_division_potentional:
                    cell = RegularTumorCell(i, j,\
                        self.cell_cycle_time, self.regular_cell_division_potentional)
                    self.cell_matrix[i, j] = cell
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

    def get_tumor_neighbors(self, i: int, j: int):
        """ Identify all adjacent tumor cells around a given cell.

        Args:
            i (int): Row index of the current cell.
            j (int): Column index of the current cell.

        Returns:
            List[Tuple[int, int]]: A list of (row, column) tuples representing
            neighboring grid positions that contain tumor cells.
        """

        return [(ni, nj) for ni, nj in self.grid.get_neighbours(i, j)
            if isinstance(self.cell_matrix[ni, nj], \
                           (RegularTumorCell, StemTumorCell, TrueStemCell))]

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
            self.immune_cells.discard(pos)
        elif isinstance(cell_obj, NecroticCell):
            self.grid[i, j] = -1
            self.active_cells.discard(pos)
            self.immune_cells.discard(pos)
        elif isinstance(cell_obj, TrueStemCell):
            self.grid[i, j] = 1
            self.active_cells.add(pos)
            self.immune_cells.discard(pos)
        elif isinstance(cell_obj, StemTumorCell):
            self.grid[i, j] = 2
            self.active_cells.add(pos)
            self.immune_cells.discard(pos)
        elif isinstance(cell_obj, RegularTumorCell):
            self.grid[i, j] = min(cell_obj.divisions_left + 2,\
                        self.regular_cell_division_potentional + 2)

            self.active_cells.add(pos)
            self.immune_cells.discard(pos)
        elif isinstance(cell_obj, ImmuneCell):
            self.grid[i, j] = -2
            self.active_cells.discard(pos)
            self.immune_cells.add(pos)

        self.cell_matrix[i, j] = cell_obj

    def __introduce_immune_cells(self):
        """Introduce immune cells at the edge of the tumor.
        """

        tumor_edge = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if isinstance(self.cell_matrix[i, j], EmptyCell):
                    neighbors = [(ni, nj) for ni, nj in self.grid.get_neighbours(i, j)]
                    for ni, nj in neighbors:
                        if isinstance(self.cell_matrix[ni, nj],\
                            (RegularTumorCell, StemTumorCell, TrueStemCell)):
                            tumor_edge.append((i, j))
                            break

        if tumor_edge:
            if len(tumor_edge) > self.immune_cell_count:
                positions = random.sample(tumor_edge, self.immune_cell_count)
            else:
                positions = tumor_edge

            for i, j in positions:
                immune_cell = ImmuneCell(
                    i, j,
                    self.cell_cycle_time,
                    self.immune_cell_lifespan,
                    self.immune_kill_probability,
                    activation_level=1.0
                )
                immune_cell.divisions_left = self.immune_cell_division_potential
                self.update_grid_cell(i, j, immune_cell)

    def __run_step(self):
        """ Executes a single time step of the tumor simulation. 
        """

        # Activate immune system if it's time
        if self.enable_immune_system and self.current_step == self.immune_activation_step:
            self.__introduce_immune_cells()

        # Get current probabilities
        probabilities = self.__get_current_probabilities()

        # Generate random values
        random_values = self.__generate_random_values()

        # Update timers for all cells
        self.__update_cell_timers()

        # Process active tumor cells
        self.__process_tumor_cells(probabilities, random_values)

        # Process immune cells
        if probabilities["immune_activation"] > 0:
            self.__process_immune_cells(probabilities, random_values)

        # Record statistics if needed
        if self.run_statistics and self.current_step % self.statistics_step == 0:
            self.__record_statistics()

        # Update time counters
        self.current_step += 1
        self.current_time += self.time_step

        # Print progress
        if self.current_step % 100 == 0:
            print(f"{self.current_step}/{self.maximum_time_steps}")

        return self.current_step <= self.maximum_time_steps

    def __get_current_probabilities(self):
        """ Extract current probability values from vectors. 
        """

        return {
            "death_prob": self.vect_deat[self.current_step],
            "prolif_prob": self.vect_prol[self.current_step],
            "migrate_prob": self.vect_potm[self.current_step],
            "stem_prob": self.vect_stem[self.current_step],
            "immune_activation": self.vect_immune[self.current_step]
        }

    def __generate_random_values(self):
        """ Generate random values for all cells. 
        """

        return {
            "tumor": {
                "death": {pos: random.random() for pos in self.active_cells},
                "prolif": {pos: random.random() for pos in self.active_cells},
                "migrate": {pos: random.random() for pos in self.active_cells}
            },
            "immune": {
                "prolif": {pos: random.random() for pos in self.immune_cells},
                "migrate": {pos: random.random() for pos in self.immune_cells},
                "kill": {pos: random.random() for pos in self.immune_cells}
            }
        }

    def __update_cell_timers(self):
        """ Update timers for all cells. 
        """

        for pos in list(self.active_cells):
            i, j = pos
            cell = self.cell_matrix[i, j]
            if cell.is_alive:
                cell.update_timer(self.time_step)

        for pos in list(self.immune_cells):
            i, j = pos
            cell = self.cell_matrix[i, j]
            if cell.is_alive:
                cell.update_timer(self.time_step)

    def __process_tumor_cells(self, probabilities, random_values):
        """ Process behavior of tumor cells. """
        death_prob = probabilities["death_prob"]
        prolif_prob = probabilities["prolif_prob"]
        migrate_prob = probabilities["migrate_prob"]
        stem_prob = probabilities["stem_prob"]

        death_random_values = random_values["tumor"]["death"]
        prolif_random_values = random_values["tumor"]["prolif"]
        migrate_random_values = random_values["tumor"]["migrate"]

        for pos in list(self.active_cells):
            i, j = pos
            cell = self.cell_matrix[i, j]

            if not cell.is_alive:
                continue

            # Process death
            if self.__process_tumor_cell_death(i, j, cell, death_prob, death_random_values[pos]):
                continue

            # Process division
            self.__process_tumor_cell_division(i, j, cell, prolif_prob, prolif_random_values[pos], stem_prob)

            # Process migration
            self.__process_tumor_cell_migration(i, j, cell, migrate_prob, migrate_random_values[pos])

    def __process_tumor_cell_death(self, i, j, cell, death_prob, random_value):
        """ Process death for a tumor cell. Returns True if cell died. """
        if random_value < death_prob:
            if not isinstance(cell, TrueStemCell) and not isinstance(cell, StemTumorCell):
                self.update_grid_cell(i, j, NecroticCell(i, j))
                self.necrotic_count += 1
                return True
        return False

    def __process_tumor_cell_division(self, i, j, cell, prolif_prob, random_value, stem_prob):
        """ Process division for a tumor cell. 
        """

        if cell.can_divide() and random_value < prolif_prob:
            empty_neighbors = self.get_empty_neighbors(i, j)
            if empty_neighbors:
                ni, nj = random.choice(empty_neighbors)
                cell.reset_timer()

                if isinstance(cell, RegularTumorCell):
                    cell.divisions_left -= 1
                    if cell.divisions_left <= 3:
                        self.update_grid_cell(i, j, NecroticCell(i, j))
                        return

                daughter = cell.divide(
                    stem_prob,
                    self.regular_cell_division_potentional,
                    self.stem_cell_potentional
                )
                self.update_grid_cell(ni, nj, daughter)

    def __process_tumor_cell_migration(self, i, j, cell, migrate_prob, random_value):
        """ Process migration for a tumor cell. 
        """

        if random_value < migrate_prob:
            empty_neighbors = self.get_empty_neighbors(i, j)
            new_pos = cell.migrate(empty_neighbors)
            if new_pos:
                ni, nj = new_pos
                self.update_grid_cell(i, j, SHARED_EMPTY_CELL)
                cell.x, cell.y = ni, nj
                self.update_grid_cell(ni, nj, cell)

    def __process_immune_cells(self, probabilities, random_values):
        """ Process behavior of immune cells. 
        """

        prolif_prob = probabilities["prolif_prob"]
        migrate_prob = probabilities["migrate_prob"]
        immune_activation = probabilities["immune_activation"]

        immune_prolif_random_values = random_values["immune"]["prolif"]
        immune_migrate_random_values = random_values["immune"]["migrate"]
        immune_kill_random_values = random_values["immune"]["kill"]

        for pos in list(self.immune_cells):
            i, j = pos
            cell = self.cell_matrix[i, j]

            if not cell.is_alive:
                self.update_grid_cell(i, j, SHARED_EMPTY_CELL)
                continue

            # Process immune cell attack
            self.__process_immune_cell_attack(i, j, cell, immune_activation, immune_kill_random_values[pos])

            # Process immune cell division
            self.__process_immune_cell_division(i, j, cell, prolif_prob, immune_prolif_random_values[pos])

            # Process immune cell migration
            self.__process_immune_cell_migration(i, j, cell, migrate_prob, immune_migrate_random_values[pos])

    def __process_immune_cell_attack(self, i, j, cell, immune_activation, random_value):
        """ Process attack behavior for an immune cell. 
        """

        tumor_neighbors = self.get_tumor_neighbors(i, j)

        if tumor_neighbors and random_value < cell.kill_probability * immune_activation:
            ti, tj = random.choice(tumor_neighbors)
            target_cell = self.cell_matrix[ti, tj]

            kill_prob = cell.kill_probability
            if isinstance(target_cell, TrueStemCell):
                kill_prob *= 0.5
            elif isinstance(target_cell, StemTumorCell):
                kill_prob *= 0.8

            if random.random() < kill_prob:
                self.update_grid_cell(ti, tj, NecroticCell(ti, tj))
                self.killed_by_immune_count += 1

    def __process_immune_cell_division(self, i, j, cell, prolif_prob, random_value):
        """ Process division for an immune cell. 
        """

        if cell.can_divide() and random_value < prolif_prob:
            empty_neighbors = self.get_empty_neighbors(i, j)
            if empty_neighbors:
                ni, nj = random.choice(empty_neighbors)
                cell.reset_timer()
                if cell.divisions_left > 0:
                    daughter = cell.divide()
                    if daughter:
                        self.update_grid_cell(ni, nj, daughter)

    def __process_immune_cell_migration(self, i, j, cell, migrate_prob, random_value):
        """ Process migration for an immune cell. 
        """

        if random_value < migrate_prob * self.immune_migration_factor:
            empty_neighbors = self.get_empty_neighbors(i, j)
            weighted_neighbors = []

            for ni, nj in empty_neighbors:
                weight = 1
                for nni, nnj in self.grid.get_neighbours(ni, nj):
                    if isinstance(self.cell_matrix[nni, nnj],\
                            (RegularTumorCell, StemTumorCell, TrueStemCell)):
                        weight = 3
                        break
                weighted_neighbors.extend([(ni, nj)] * weight)

            if weighted_neighbors:
                ni, nj = random.choice(weighted_neighbors)
                self.update_grid_cell(i, j, SHARED_EMPTY_CELL)
                cell.x, cell.y = ni, nj
                self.update_grid_cell(ni, nj, cell)

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
        total cells, tumor cells, stem cells, immune cells and necrotic cells.
        """

        total, tumor, stem, true_stem, immune_count = 0, 0, 0, 0, 0

        for pos in self.active_cells:
            i, j = pos
            cell = self.cell_matrix[i, j]
            total += 1
            if isinstance(cell, TrueStemCell):
                true_stem += 1
                stem += 1
                tumor += 1
            if isinstance(cell, StemTumorCell):
                stem += 1
                tumor += 1
            elif isinstance(cell, RegularTumorCell):
                tumor += 1

        for pos in self.immune_cells:
            i, j = pos
            cell = self.cell_matrix[i, j]
            if cell.is_alive:
                immune_count += 1
                total += 1

        current_day = int(self.current_time)
        if current_day == 0 or current_day >= self.last_plot_day + self.plot_interval:
            self._visualize_grid()
            self.last_plot_day = current_day

        self.stats['time'][self.stats_index] = self.current_time
        self.stats['total_cells'][self.stats_index] = total + self.necrotic_count
        self.stats['true_stems'][self.stats_index] = true_stem
        self.stats['tumor_cells'][self.stats_index] = tumor
        self.stats['stem_cells'][self.stats_index] = stem
        self.stats['necrotic_cells'][self.stats_index] = self.necrotic_count
        self.stats['immune_cells'][self.stats_index] = immune_count
        self.stats['killed_tumor_cells'][self.stats_index] = self.killed_by_immune_count

        self.stats_index += 1

    def _visualize_grid(self):
        """
        Generates and saves a visualization of the cell grid, showing different cell types
        with distinct colors.
        """
        # Create a proper color mapping for all cell types
        # -2: Immune cells (blue)
        # -1: Necrotic cells (orange)
        # 0: Empty cells (white/light gray)
        # 1: True stem cells (yellow)
        # 2: Stem tumor cells (green)
        # 3-11: Regular tumor cells (red gradient)

        colors = ['#0000ff',
                '#fdae61',
                '#dddddd']

        colors.append('#fff426')

        colors.append('#00ff00')

        red_grad = list(map(lambda x: '#' + format(int(x), '02x') + '0000',
                        np.linspace(70, 255, self.regular_cell_division_potentional)))
        colors.extend(red_grad)

        cmap = ListedColormap(colors, name='cell_types')

        vis_grid = self.grid.map.copy() + 2

        bounds = np.arange(len(colors) + 1)
        norm = BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(14, 8))
        plt.matshow(vis_grid, cmap=cmap, norm=norm)

        legend_elements = [
            Patch(facecolor='#0000ff', edgecolor='black', label='Immune Cell'),
            Patch(facecolor='#fdae61', edgecolor='black', label='Necrotic Cell'),
            Patch(facecolor='#dddddd', edgecolor='black', label='Empty Cell'),
            Patch(facecolor='#fff426', edgecolor='black', label='True Stem Cell'),
            Patch(facecolor='#00ff00', edgecolor='black', label='Stem Tumor Cell'),
            Patch(facecolor=red_grad[0], edgecolor='black', label='Regular Tumor Cell (Low Division Potential)'),
            Patch(facecolor=red_grad[-1], edgecolor='black', label='Regular Tumor Cell (High Division Potential)')
        ]

        plt.legend(
            handles=legend_elements,
            loc='center right',
            bbox_to_anchor=(-0.1, 0.5),
            prop={'size': 10}
        )

        plt.colorbar(ticks=bounds[:-1] + 0.5, label='Cell Type')
        plt.title(f'Tumor Simulation - Day: {int(self.current_time)}')
        plt.savefig(f'tumor_day_{int(self.current_time)}.png', bbox_inches='tight')
        plt.close()

    def plot_statistics(self):
        """ Plots the evolution of key statistics over time.
        """

        plt.figure(figsize=(12, 8))

        t = self.stats['time'][:self.stats_index]

        plt.plot(t, self.stats['total_cells'][:self.stats_index], 'k-', label='Total Cells', linewidth=2)
        plt.plot(t, self.stats['tumor_cells'][:self.stats_index], 'r-', label='Tumor Cells', linewidth=2)
        plt.plot(t, self.stats['stem_cells'][:self.stats_index], 'g--', label='Stem Cells', linewidth=2.5)
        plt.plot(t, self.stats['true_stems'][:self.stats_index], 'y-o', label='True Stem Cells', markersize=4, linewidth=2)
        plt.plot(t, self.stats['necrotic_cells'][:self.stats_index], 'b-.', label='Necrotic Cells', linewidth=2)
        plt.plot(t, self.stats['immune_cells'][:self.stats_index], 'm-', label='Immune Cells', linewidth=2)
        plt.plot(t, self.stats['killed_tumor_cells'][:self.stats_index], 'c--', label='Killed by Immune', linewidth=2)

        plt.xlabel('Time (days)', fontsize=14)
        plt.ylabel('Cell Count', fontsize=14)
        plt.title('Tumor Growth Simulation with Immune Response', fontsize=16)

        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('tumor_stats.png')
        return plt.gcf()

if __name__ == "__main__":
    sim = TumorSimulation(
        grid_size=(1500, 1500),
        max_simulation_days=100,
        time_step=1/12,
        cell_cycle_time=24,
        regular_cell_division_potentional=11,
        stem_cell_potentional=12,
        migration_potentional=1,
        tumor_size=100,
        immune_activation_day=10,
        immune_cell_count=20,
        immune_cell_lifespan=72,
        immune_kill_probability=0.3,
        immune_migration_factor=2.0,
        enable_immune_system=True,
        run_statistics=True,
        statistics_step=100
    )

    sim.run_simulation()
