# Tumor Cellular Automaton Model

## Table of Contents

* [Overview](#overview)
* [Theoretical Background](#theoretical-background)

  * [Cellular Automata in Cancer Modeling](#cellular-automata-in-cancer-modeling)
  * [Tumor Cell Hierarchy](#tumor-cell-hierarchy)
  * [Tumor Growth Dynamics](#tumor-growth-dynamics)
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

This project simulates tumor development using a **stochastic cellular automaton (CA)** model. It captures essential biological behaviors such as cell proliferation, migration, differentiation, and death. Special emphasis is placed on **cancer stem cell dynamics**, providing a framework to study tumor heterogeneity and growth.

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

This structure models clonal evolution and the differentiation hierarchy found in real tumors.

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

---

## Implementation Details

### Architecture Overview

* **Grid**: 2D lattice tracking spatial cell layout and neighborhood relationships
* **Cell**: Encapsulates behavior and type-specific logic (TSC, STC, RTC, Necrotic)
* **TumorSimulation**: Governs time evolution, probabilistic behavior, and data collection

### Key Parameters

| Parameter                          | Description                             |
| ---------------------------------- | --------------------------------------- |
| `grid_size`                        | Dimensions of the simulation grid       |
| `max_simulation_days`              | Total simulated time in days            |
| `time_step`                        | Time step granularity (in days)         |
| `cell_cycle_time`                  | Division time (in hours)                |
| `regular_cell_division_potential`  | Max divisions per RTC                   |
| `stem_cell_potential`              | Unlimited or capped stem cell divisions |
| `tumor_size`                       | Initial tumor seed radius               |
| `TSC_prob`, `STC_prob`, `RST_prob` | Probabilities for initial cell types    |

### Simulation Workflow

At each time step:

1. For every active cell:

   * Update division timers
   * Check for death
   * Attempt division (if eligible)
   * Attempt migration

2. All behaviors are governed by probabilities:

   * `vect_deat[t]`: Death
   * `vect_prol[t]`: Division
   * `vect_potm[t]`: Migration

3. Statistics are recorded at user-defined intervals

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
python main.py --n 1000 --m 1000 --days 200 --div_pot 15 --tumor_size 50 -record_stats
```

### Command-Line Options

| Flag              | Description                                  | Default |
| ----------------- | -------------------------------------------- | ------- |
| `--n`             | Grid rows                                    | 500     |
| `--m`             | Grid columns                                 | 500     |
| `--seed`          | RNG seed for reproducibility                 | None    |
| `--days`          | Duration in simulation days                  | 300     |
| `--div_pot`       | Division potential of RTCs                   | 15      |
| `--tumor_size`    | Initial tumor radius                         | 50      |
| `--stat_step`     | Interval for statistics recording            | 10      |
| `-record_stats`   | Enable statistics recording                  | False   |
| `--plot_interval` | Time between saved visualizations (in steps) | 100     |

---

## Output and Example Results

When `-record_stats` is enabled, the following are saved:

* **Tumor Snapshots**: Images showing spatial layout of tumor cells over time
* **Population Plots**: Graphs tracking TSC, STC, RTC, and necrotic cell counts

All output files are stored in the current working directory.

---

## Contributors

* **Yaryna Pechenenko** – Implemented the cell classes and cell type logic
* **Ivan Zarytskyi** – Developed the visualization tools and output plotting
* **Mykhailo Rykhalskyi** – Designed and implemented the grid structure and spatial logic
* **Roman Prokhorov** – Built the main simulation engine and time evolution mechanics

## Supervisor

* **Max Bug**
