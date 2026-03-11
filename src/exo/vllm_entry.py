from __future__ import annotations

from exo.vllm_patches.growable_cache import patch_vllm

patch_vllm()

if __name__ == "__main__":
    import sys

    import anyio
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(sys.argv[1:])
    anyio.run(run_server, args)
