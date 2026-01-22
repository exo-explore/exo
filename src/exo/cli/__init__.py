"""Exo CLI - SLURM-compatible job management commands."""


def run_subcommand(command: str, args: list[str]) -> int:
    """Route to the appropriate subcommand handler.

    Args:
        command: The subcommand name (sbatch, squeue, scancel, salloc)
        args: Command line arguments for the subcommand

    Returns:
        Exit code from the subcommand
    """
    if command == "sbatch":
        from exo.cli.sbatch import main

        return main(args)
    elif command == "squeue":
        from exo.cli.squeue import main

        return main(args)
    elif command == "scancel":
        from exo.cli.scancel import main

        return main(args)
    elif command == "salloc":
        from exo.cli.salloc import main

        return main(args)
    else:
        print(f"Unknown subcommand: {command}")
        return 1
