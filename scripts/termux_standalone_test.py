#!/usr/bin/env python3
"""
Standalone llama.cpp test for Termux/Android
=============================================
This script tests llama.cpp inference WITHOUT the full exo framework.
Use this to verify your Android device can run LLM inference.

Usage:
    python3 scripts/termux_standalone_test.py
    python3 scripts/termux_standalone_test.py --model ~/.exo/models/path/to/model.gguf
    python3 scripts/termux_standalone_test.py --interactive
"""

import argparse
import sys
import time
from pathlib import Path


def get_device_info() -> dict:
    """Get Android device information."""
    info = {"platform": "unknown", "ram_gb": 0, "cpu_cores": 0}
    
    try:
        import platform
        info["platform"] = f"{platform.system()} {platform.machine()}"
        info["python"] = platform.python_version()
    except Exception:
        pass
    
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        pass
    
    try:
        import os
        info["cpu_cores"] = os.cpu_count() or 1
    except Exception:
        pass
    
    return info


def find_model() -> Path | None:
    """Find a GGUF model in common locations."""
    search_paths = [
        Path.home() / ".exo" / "models",
        Path.home() / "models",
        Path("/sdcard/models"),
        Path.cwd(),
    ]
    
    for base in search_paths:
        if base.exists():
            for gguf in base.rglob("*.gguf"):
                return gguf
    return None


def run_inference(model_path: Path, prompt: str, max_tokens: int = 64) -> str:
    """Run inference using llama-cpp-python."""
    from llama_cpp import Llama
    
    print(f"\n📂 Loading model: {model_path.name}")
    print(f"   Size: {model_path.stat().st_size / (1024**2):.1f} MB")
    
    start_time = time.time()
    
    # Conservative settings for Android
    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,
        n_threads=4,  # Don't use all cores, save some for system
        n_gpu_layers=0,  # CPU only
        verbose=False,
    )
    
    load_time = time.time() - start_time
    print(f"   Loaded in {load_time:.1f}s")
    
    print(f"\n🤖 Generating response...")
    print(f"   Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    
    start_time = time.time()
    
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["\n\n"],
        echo=False,
    )
    
    gen_time = time.time() - start_time
    response = output["choices"][0]["text"].strip()
    
    # Calculate tokens per second
    tokens_generated = output.get("usage", {}).get("completion_tokens", len(response.split()))
    tps = tokens_generated / gen_time if gen_time > 0 else 0
    
    print(f"\n💬 Response:\n{response}")
    print(f"\n📊 Stats:")
    print(f"   Tokens: {tokens_generated}")
    print(f"   Time: {gen_time:.2f}s")
    print(f"   Speed: {tps:.1f} tokens/sec")
    
    return response


def interactive_mode(model_path: Path):
    """Run interactive chat mode."""
    from llama_cpp import Llama
    
    print(f"\n📂 Loading model for interactive chat...")
    
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False,
    )
    
    print("✅ Model loaded! Type 'quit' to exit.\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            
            # Simple prompt format
            full_prompt = f"User: {prompt}\nAssistant:"
            
            output = llm(
                full_prompt,
                max_tokens=256,
                temperature=0.7,
                stop=["User:", "\n\n"],
            )
            
            response = output["choices"][0]["text"].strip()
            print(f"AI: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone llama.cpp test for Termux"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive chat mode"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Hello! Please introduce yourself in one sentence.",
        help="Prompt to send to the model"
    )
    args = parser.parse_args()
    
    print("=" * 50)
    print("  Termux llama.cpp Standalone Test")
    print("=" * 50)
    
    # Show device info
    print("\n📱 Device Info:")
    info = get_device_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Check llama-cpp-python
    print("\n🔍 Checking llama-cpp-python...")
    try:
        import llama_cpp
        print(f"   ✅ Version: {llama_cpp.__version__}")
    except ImportError as e:
        print(f"   ❌ Not installed: {e}")
        print("\n   Install with: pip install llama-cpp-python --no-cache-dir")
        sys.exit(1)
    
    # Find model
    model_path = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"\n❌ Model not found: {model_path}")
            sys.exit(1)
    else:
        print("\n🔍 Searching for models...")
        model_path = find_model()
        if model_path:
            print(f"   ✅ Found: {model_path}")
        else:
            print("   ❌ No models found!")
            print("\n   Download a model first:")
            print("   ./scripts/download_model.sh qwen-0.5b")
            sys.exit(1)
    
    # Run test
    try:
        if args.interactive:
            interactive_mode(model_path)
        else:
            run_inference(model_path, args.prompt)
            print("\n" + "=" * 50)
            print("  ✅ SUCCESS! llama.cpp works on your device!")
            print("=" * 50)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

