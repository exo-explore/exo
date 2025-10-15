#!/usr/bin/env python3
"""
Launcher for MoE distributed test that handles environment setup
"""

import os
import sys
from pathlib import Path

# Ensure we're in the right directory
os.chdir(Path(__file__).parent)

# Add exo to path
sys.path.insert(0, str(Path(__file__).parent))

# Now import and run the test
if __name__ == "__main__":
    # Parse arguments
    args = sys.argv[1:]
    
    # Import the test module
    import test_moe_distributed
    
    # Add arguments back to sys.argv for argparse
    sys.argv = ["test_moe_distributed.py"] + args
    
    # Run main
    test_moe_distributed.main()