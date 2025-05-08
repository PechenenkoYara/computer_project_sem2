# Tumor Cellular Automaton Model

## Table of Contents

* [Overview](#overview)
* [Theoretical Background](#theoretical-background)
  * [Cellular Automata in Cancer Modeling](#cellular-automata-in-cancer-modeling)
  * [Tumor Cell Hierarchy](#tumor-cell-hierarchy)
  * [Tumor Growth Dynamics](#tumor-growth-dynamics)
  * [Immune System Interaction](#immune-system-interaction)
* [Implementation Details](#implementation-details)
  * [Architecture Overview](#architecture-overview)
  * [Key Parameters](#key-parameters)
  * [Simulation Workflow](#simulation-workflow)
* [Installation](#installation)
* [Usage](#usage)
* [Output and Example Results](#output-and-example-results)
* [Contributors](#contributors)
* [Supervisor](#supervisor)

---

## Overview

This project simulates tumor development using a **stochastic cellular automaton (CA)** model. It captures essential biological behaviors such as cell proliferation, migration, differentiation, and death. Special emphasis is placed on **cancer stem cell dynamics** and **immune system interactions**, providing a framework to study tumor heterogeneity, growth, and immune response.

---

## Theoretical Background

### Cellular Automata in Cancer Modeling

Cellular automata are well-suited for modeling complex, spatially distributed systems such as tumors. Key features include:

* A discrete 2D lattice representing tissue structure
* Local rules driving cell behavior based on neighboring states
* Time progressing in discrete steps
* Simple rules leading to emergent, complex tumor behavior

Advantages:

* Simulates spatial heterogeneity and invasion patterns
* Captures inter-cellular competition
* Allows visualization of tumor growth in space and time

### Tumor Cell Hierarchy

The model implements a biologically inspired hierarchy:

1. **True Stem Cells (TSC)**

   * Unlimited division
   * Capable of self-renewal or differentiation into STCs
   * Responsible for long-term growth and relapse

2. **Stem Tumor Cells (STC)**

   * Unlimited division
   * Can only produce Regular Tumor Cells

3. **Regular Tumor Cells (RTC)**

   * Limited proliferative potential (set by `regular_cell_division_potential`)
   * Eventually die and become necrotic

4. **Necrotic Cells**

   * Inactive remnants of dead RTCs

5. **Immune Cells**

   * Can target and kill tumor cells
   * Limited lifespan and division potential
   * Actively migrate toward tumor cells

This structure models clonal evolution, the differentiation hierarchy found in real tumors, and the immune system's natural response.

### Tumor Growth Dynamics

Each cell may undergo the following events:

* **Proliferation**:
  Division occurs based on cell cycle timing and `vect_prol` probability.

* **Migration**:
  Cells can move into neighboring empty grid spaces, with probability `vect_potm`.

* **Death**:
  Cells die either when their division limit is reached (RTC) or stochastically via `vect_deat`.

* **Stem Cell Fate Decisions**:
  TSCs choose between self-renewal and differentiation according to `vect_stem`.

### Immune System Interaction

The immune system component models how host immunity responds to and interacts with growing tumors:

* **Activation Timing**:
  Immune response begins at a specified day during simulation.

* **Cell Deployment**:
  Immune cells are introduced at the tumor periphery.

* **Killing Mechanics**:
  Immune cells can eliminate tumor cells with varying probabilities based on cell type:
  - Regular tumor cells: Base kill probability
  - Stem tumor cells: Reduced kill probability (80% of base)
  - True stem cells: Further reduced kill probability (50% of base)

* **Directed Migration**:
  Immune cells preferentially move toward areas with tumor cells.

* **Proliferation**:
  Immune cells can divide a limited number of times to model clonal expansion.

* **Lifespan**:
  Immune cells remain active for a defined period before disappearing.

---

## Implementation Details

### Architecture Overview

* **Grid**: 2D lattice tracking spatial cell layout and neighborhood relationships
* **Cell**: Encapsulates behavior and type-specific logic (TSC, STC, RTC, Necrotic, Immune)
* **TumorSimulation**: Governs time evolution, probabilistic behavior, and data collection

### Key Parameters

| Parameter                          | Description                               |
| ---------------------------------- | ----------------------------------------- |
| `grid_size`                        | Dimensions of the simulation grid         |
| `max_simulation_days`              | Total simulated time in days              |
| `time_step`                        | Time step granularity (in days)           |
| `cell_cycle_time`                  | Division time (in hours)                  |
| `regular_cell_division_potential`  | Max divisions per RTC                     |
| `stem_cell_potential`              | Unlimited or capped stem cell divisions   |
| `tumor_size`                       | Initial tumor seed radius                 |
| `TSC_prob`, `STC_prob`, `RST_prob` | Probabilities for initial cell types      |
| `immune_activation_day`            | Day when immune response begins           |
| `immune_cell_count`                | Initial number of immune cells introduced |
| `immune_cell_lifespan`             | Survival time of immune cells (hours)     |
| `immune_cell_division_potential`   | Division capacity of immune cells         |
| `immune_kill_probability`          | Base probability of killing tumor cells   |
| `immune_migration_factor`          | Movement rate multiplier for immune cells |
| `enable_immune_system`             | Toggle immune system functionality        |

### Simulation Workflow

At each time step:

1. For every active tumor cell:

   * Update division timers
   * Check for death
   * Attempt division (if eligible)
   * Attempt migration

2. For every immune cell (if immune system is enabled):

   * Update lifespan timers
   * Target and attempt to kill neighboring tumor cells
   * Attempt division (if eligible)
   * Migrate (preferentially toward tumor cells)

3. All behaviors are governed by probabilities:

   * `vect_deat[t]`: Death
   * `vect_prol[t]`: Division
   * `vect_potm[t]`: Migration
   * `vect_immune[t]`: Immune activity level

4. Statistics are recorded at user-defined intervals

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/PechenenkoYara/computer_project_sem2.git
cd computer_project_sem2
```
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

Run a simulation with default settings:

```bash
python main.py
```

Run with custom parameters:

```bash
python main.py --n 1000 --m 1000 --days 200 --div_pot 15 --tumor_size 50 --immune --immune_day 10 --immune_count 20 --record_stats
```

### Command-Line Options

| Flag                  | Description                                          | Default |
| --------------------- | ---------------------------------------------------- | ------- |
| `--n`                 | Grid rows                                            | 500     |
| `--m`                 | Grid columns                                         | 500     |
| `--seed`              | RNG seed for reproducibility                         | None    |
| `--days`              | Duration in simulation days                          | 100     |
| `--time_step`         | Time step in days                                    | 1/12    |
| `--cell_cycle`        | Cell cycle time in hours                             | 24      |
| `--div_pot`           | Division potential of RTCs                           | 11      |
| `--stem_pot`          | Division potential for stem cells                    | 12      |
| `--migration`         | Migration potential                                  | 1       |
| `--tumor_size`        | Initial tumor radius                                 | 10      |
| `--tsc_prob`          | True stem cell probability                           | 0.1     |
| `--stc_prob`          | Stem tumor cell probability                          | 0.2     |
| `--rst_prob`          | Regular stem tumor cell probability                  | 0.5     |
| `--immune`            | Enable immune system                                 | False   |
| `--immune_day`        | Day when immune response starts                      | 5       |
| `--immune_count`      | Number of immune cells to introduce                  | 10      |
| `--immune_lifespan`   | Immune cell lifespan in hours                        | 72      |
| `--immune_div`        | Immune cell division potential                       | 3       |
| `--immune_kill`       | Probability for immune cells to kill tumor cells     | 0.3     |
| `--immune_migration`  | Migration factor for immune cells                    | 2.0     |
| `--record_stats`      | Enable statistics recording                          | False   |
| `--stat_step`         | Interval for statistics recording                    | 10      |
| `--plot_interval`     | Time between saved visualizations (in days)          | 100     |

---

## Output and Example Results

When `--record_stats` is enabled, the following are saved:

* **Tumor Snapshots**: Images showing spatial layout of tumor and immune cells over time
* **Population Plots**: Graphs tracking TSC, STC, RTC, necrotic cell, and immune cell counts
* **Immune Response Data**: Statistics on tumor cells killed by immune cells

All output files are stored in the current working directory. Visualization images are named according to the simulation day they represent (e.g., `tumor_day_10.png`).

The color scheme for visualizations is:
- Blue: Immune cells
- Yellow: True stem cells
- Green: Stem tumor cells
- Red gradient: Regular tumor cells (darker = higher division potential)
- Orange: Necrotic cells
- Light gray: Empty spaces

---

## Contributors

* **Yaryna Pechenenko** – Implemented the cell classes and cell type logic
* **Ivan Zarytskyi** – Developed the visualization tools, cli and output plotting
* **Mykhailo Rykhalskyi** – Designed and implemented the grid structure and spatial logic
* **Roman Prokhorov** – Built the main simulation engine and time evolution mechanics

## Supervisor

* **Max Bug**
