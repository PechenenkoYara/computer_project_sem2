"""
Main Module
"""

import argparse
import simulation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the tumor simulation.')
    parser.add_argument(
        '--n',
        type=int,
        default=500,
        help='Number of rows in the grid (default: 500)'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=500,
        help='Number of columns in the grid (default: 500)'
    )
    parser.add_argument(
        '--seed',
        type=str,
        default=None,
        help='Seed for random number generation (default: None)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=300,
        help='Number of days to simulate (default: 300)'
    )
    parser.add_argument(
        '--div_pot',
        type=int,
        default=15,
        help='Number of times a cell can divide (default: 15)'
    )
    parser.add_argument(
        '--tumor_size',
        type=int,
        default=50,
        help='Initial tumor size (default: 50)'
    )
    parser.add_argument(
        '--stat_step',
        type=int,
        default=10,
        help='Step size for statistics (default: 10)'
    )
    parser.add_argument(
        '-record_stats',
        action='store_true',
        help='Record statistics (default: False)'
    )
    parser.add_argument(
        '--plot_interval',
        type=int,
        default=100,
        help='Interval of days between visualization plots (default: 100)'
    )

    args = parser.parse_args()
    # print(args)
    sim = simulation.TumorSimulation(
        grid_size=(args.n, args.m), 
        max_simulation_days=args.days, 
        regular_cell_division_potentional=args.div_pot, 
        tumor_size=args.tumor_size, 
        statistics_step=args.stat_step, 
        run_statistics=args.record_stats, 
        seed=args.seed,
        plot_interval=args.plot_interval
    )
    sim.run_simulation()
