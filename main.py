"""
Main Module for Tumor Simulation

This module provides a command-line interface to run tumor growth simulations
with immune system interactions.
"""

import argparse
import simulation

def parse_arguments():
    """Parse command line arguments for the tumor simulation.
    """

    parser = argparse.ArgumentParser(description='Run tumor growth simulation with immune system interaction.')

    parser.add_argument('--n',
        type=int,
        default=500,
        help='Number of rows in the grid (default: 500)'
    )

    parser.add_argument('--m',
        type=int,
        default=500,
        help='Number of columns in the grid (default: 500)'
    )

    parser.add_argument('--days',
        type=int,
        default=100,
        help='Number of days to simulate (default: 100)'
    )

    parser.add_argument('--time_step',
        type=float,
        default=1/12,
        help='Time step in days (default: 1/12, i.e., 2 hours)'
    )

    parser.add_argument('--seed',
        type=str,
        default=None,
        help='Seed for random number generation (default: None)'
    )

    # Cell behavior parameters
    parser.add_argument('--cell_cycle',
        type=int,
        default=24,
        help='Cell cycle time in hours (default: 24)'
    )

    parser.add_argument('--div_pot',
        type=int,
        default=11,
        help='Division potential for regular tumor cells (default: 11)'
    )

    parser.add_argument('--stem_pot',
        type=int,
        default=12,
        help='Division potential for stem cells (default: 12)'
    )

    parser.add_argument('--migration',
        type=float,
        default=1,
        help='Migration potential (default: 1)'
    )

    # Initial tumor parameters
    parser.add_argument('--tumor_size',
        type=int,
        default=10,
        help='Initial tumor size (default: 10)'
    )

    parser.add_argument('--tsc_prob',
        type=float,
        default=0.1,
        help='True stem cell probability (default: 0.1)'
    )

    parser.add_argument('--stc_prob',
        type=float,
        default=0.2,
        help='Stem tumor cell probability (default: 0.2)'
    )

    parser.add_argument('--rst_prob',
        type=float,
        default=0.5,
        help='Regular stem tumor cell probability (default: 0.5)'
    )

    # Immune system parameters
    parser.add_argument('--immune',
        action='store_false',
        help='Enable immune system (default: False)'
    )

    parser.add_argument('--immune_day',
        type=int,
        default=5,
        help='Day when immune response starts (default: 5)'
    )

    parser.add_argument('--immune_count',
        type=int,
        default=10,
        help='Number of immune cells to introduce (default: 10)'
    )

    parser.add_argument('--immune_lifespan',
        type=int,
        default=72,
        help='Immune cell lifespan in hours (default: 72)'
    )

    parser.add_argument('--immune_div',
        type=int,
        default=3,
        help='Immune cell division potential (default: 3)'
    )

    parser.add_argument('--immune_kill',
        type=float,
        default=0.3,
        help='Probability for immune cells to kill tumor cells (default: 0.3)'
    )

    parser.add_argument('--immune_migration',
        type=float,
        default=2.0,
        help='Migration factor for immune cells (default: 2.0)'
    )

    # Statistics and visualization
    parser.add_argument('--record_stats',
        action='store_true',
        help='Record statistics (default: False)'
    )

    parser.add_argument('--stat_step',
        type=int,
        default=10,
        help='Step size for statistics (default: 10)'
    )

    parser.add_argument('--plot_interval',
        type=int,
        default=100,
        help='Interval of days between visualization plots (default: 100)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # Print simulation parameters
    print("Starting tumor simulation with parameters:")
    print(f"Grid size: {args.n}x{args.m}")
    print(f"Simulation length: {args.days} days")
    print(f"Initial tumor size: {args.tumor_size}")
    print(f"Immune system: {'Enabled' if args.immune else 'Disabled'}")
    if args.immune:
        print(f"Immune activation day: {args.immune_day}")
        print(f"Immune cell count: {args.immune_count}")

    # Create and run simulation
    sim = simulation.TumorSimulation(
        grid_size=(args.n, args.m),
        max_simulation_days=args.days,
        time_step=args.time_step,
        cell_cycle_time=args.cell_cycle,
        regular_cell_division_potentional=args.div_pot,
        stem_cell_potentional=args.stem_pot,
        migration_potentional=args.migration,
        tumor_size=args.tumor_size,
        TSC_prob=args.tsc_prob,
        STC_prob=args.stc_prob,
        RST_prob=args.rst_prob,

        # Immune system parameters
        immune_activation_day=args.immune_day,
        immune_cell_count=args.immune_count,
        immune_cell_lifespan=args.immune_lifespan,
        immune_cell_division_potential=args.immune_div,
        immune_kill_probability=args.immune_kill,
        immune_migration_factor=args.immune_migration,
        enable_immune_system=args.immune,

        # Other settings
        seed=args.seed,
        run_statistics=args.record_stats,
        statistics_step=args.stat_step,
        plot_interval=args.plot_interval
    )

    print("Running simulation...")
    sim.run_simulation()
    print("Simulation completed!")
