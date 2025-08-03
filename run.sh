#!/bin/bash
DIR="$PWD"

# Initialize flags
REPLICA=false
CLEAN=false

# Parse command line arguments
while getopts "rc" opt; do
  case $opt in
    r)
      REPLICA=true
      ;;
    c)
      CLEAN=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 [-r] [-c]"
      echo "  -r  Run as replica"
      echo "  -c  Clean databases before starting"
      exit 1
      ;;
  esac
done

# Clean if requested
if [ "$CLEAN" = true ]; then
  echo "Cleaning databases..."
  rm -f ~/.exo/*db*
  rm -f ~/.exo_replica/*db*
fi

# Configure MLX
./configure_mlx.sh

# First command (worker) - changes based on replica flag
if [ "$REPLICA" = true ]; then
  osascript -e "tell app \"Terminal\" to do script \"cd '$DIR'; nix develop -c bash -c 'export EXO_HOME=.exo_replica; uv run -m worker.main'\""
else
  osascript -e "tell app \"Terminal\" to do script \"cd '$DIR'; nix develop -c uv run -m worker.main\""
fi

# Second command (master) - changes based on replica flag
if [ "$REPLICA" = true ]; then
  osascript -e "tell app \"Terminal\" to do script \"cd '$DIR'; nix develop -c bash -c 'export RUST_LOG=true EXO_RUN_AS_REPLICA=1 EXO_HOME=.exo_replica API_PORT=8001; uv run -m master.main'\""
else
  osascript -e "tell app \"Terminal\" to do script \"cd '$DIR'; nix develop -c bash -c 'export RUST_LOG=true; uv run -m master.main'\""
fi