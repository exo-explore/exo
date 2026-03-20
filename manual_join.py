import os
import sys

# 1. Force the identity at the OS Level
os.environ["EXO_LIBP2P_NAMESPACE"] = "production_cluster_01"
os.environ["EXO_DISCOVERY_PEERS"] = "10.0.0.14"
os.environ["EXO_MASTER_NODE_ID"] = "12D3KooWLNxCViNXehagJi9KauAZW5bvPx2kUPLqu5P7HDmUH1Gd"

# 2. Tell the Mini it is NOT allowed to be Master
# We'll use a Python-level override
sys.argv = ['exo', '--no-api']

from exo.main import main  # noqa: E402

if __name__ == "__main__":
    # We will manually strip the Master role from the starting components
    print("🧠 Programmatic Join: Forcing Worker Role...")
    main()
