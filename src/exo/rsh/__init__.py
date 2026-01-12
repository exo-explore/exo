"""Exo RSH - Remote Shell for MPI without SSH.

This module provides a remote execution mechanism that allows mpirun to spawn
processes on remote nodes without requiring SSH setup. It works by:

1. Each Exo node runs a small HTTP server (RSH server) on port 52416
2. The exo-rsh script acts as a drop-in replacement for ssh
3. When mpirun calls "exo-rsh hostname command", it HTTP POSTs to the target
4. The target executes the command and streams output back

Usage:
    mpirun --mca plm_rsh_agent exo-rsh -np 4 --hostfile hosts.txt ./program
"""
