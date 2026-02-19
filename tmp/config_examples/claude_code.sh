#!/bin/bash
# Run Claude Code against a local exo cluster! (Here, GPT OSS 120B)
ANTHROPIC_BASE_URL="http://localhost:52415/" \
  ANTHROPIC_AUTH_TOKEN="dummy" \
  ANTHROPIC_MODEL="mlx-community/gpt-oss-120b-MXFP4-Q8" \
  ANTHROPIC_SMALL_FAST_MODEL="mlx-community/gpt-oss-120b-MXFP4-Q8" \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  claude
