#!/usr/bin/env python3
"""
Simple test script to verify llama.cpp inference works on Termux/Android.

Usage:
    python3 scripts/test_inference.py
    python3 scripts/test_inference.py --model /path/to/model.gguf
"""

import os
import sys
import glob
import argparse
from pathlib import Path


def find_model() -> Path | None:
    """Find a GGUF model in the default exo models directory."""
    models_dir = Path.home() / ".exo" / "models"
    
    if not models_dir.exists():
        return None
    
    # Search for .gguf files recursively
    for gguf in models_dir.rglob("*.gguf"):
        return gguf
    
    return None


def get_system_info() -> dict[str, str]:
    """Get basic system information."""
    info = {}
    
    try:
        import platform
        info["system"] = platform.system()
        info["machine"] = platform.machine()
        info["python"] = platform.python_version()
    except Exception:
        pass
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_gb"] = f"{mem.total / (1024**3):.1f}"
        info["ram_available_gb"] = f"{mem.available / (1024**3):.1f}"
    except Exception:
        pass
    
    return info


def test_inference(model_path: Path) -> bool:
    """Run a simple inference test."""
    print(f"\n🔄 Loading model: {model_path.name}")
    print(f"   Full path: {model_path}")
    print(f"   Size: {model_path.stat().st_size / (1024**2):.1f} MB")
    
    try:
        from llama_cpp import Llama
        
        # Load model with conservative settings for Android
        print("\n⏳ Initializing model (this may take a moment)...")
        
        llm = Llama(
            model_path=str(model_path),
            n_ctx=512,           # Small context for testing
            n_threads=4,         # Conservative thread count
            n_gpu_layers=0,      # CPU only
            verbose=False,
        )
        
        print("✓ Model loaded successfully!")
        
        # Run a simple inference
        print("\n🤖 Running test inference...")
        prompt = "Hello! Please respond with a single short sentence."
        
        output = llm(
            prompt,
            max_tokens=32,
            temperature=0.7,
            stop=["\n", "."],
        )
        
        response = output["choices"][0]["text"].strip()
        
        print(f"\n📝 Prompt: {prompt}")
        print(f"💬 Response: {response}")
        
        # Show usage stats
        if "usage" in output:
            usage = output["usage"]
            print(f"\n📊 Tokens - Prompt: {usage.get('prompt_tokens', '?')}, "
                  f"Generated: {usage.get('completion_tokens', '?')}")
        
        print("\n✅ Inference test PASSED!")
        return True
        
    except ImportError as e:
        print(f"\n❌ llama-cpp-python not installed: {e}")
        print("   Try: pip install llama-cpp-python --no-cache-dir")
        return False
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test llama.cpp inference on Termux")
    parser.add_argument("--model", "-m", type=str, help="Path to GGUF model file")
    args = parser.parse_args()
    
    print("=" * 50)
    print("   exo llama.cpp Inference Test")
    print("=" * 50)
    
    # Show system info
    print("\n📱 System Information:")
    info = get_system_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Check for llama_cpp
    print("\n🔍 Checking llama-cpp-python...")
    try:
        import llama_cpp
        print(f"   ✓ Version: {llama_cpp.__version__}")
    except ImportError:
        print("   ❌ llama-cpp-python not installed")
        print("   Run: pip install llama-cpp-python --no-cache-dir")
        sys.exit(1)
    
    # Find or use specified model
    model_path = None
    
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"\n❌ Model not found: {model_path}")
            sys.exit(1)
    else:
        print("\n🔍 Looking for models in ~/.exo/models/...")
        model_path = find_model()
        
        if model_path is None:
            print("   ❌ No models found!")
            print("\n   Download a model first:")
            print("   ./scripts/download_model.sh qwen-0.5b")
            print("   ./scripts/download_model.sh tinyllama")
            sys.exit(1)
        
        print(f"   ✓ Found: {model_path.name}")
    
    # Run inference test
    success = test_inference(model_path)
    
    if success:
        print("\n" + "=" * 50)
        print("   🎉 All tests passed! exo is ready for inference.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("   ⚠️  Some tests failed. Check the errors above.")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()

