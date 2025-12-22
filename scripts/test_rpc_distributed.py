#!/usr/bin/env python3
"""
Test script for llama.cpp distributed RPC inference.

This script tests the RPC-based distributed inference functionality:
1. Verifies rpc-server binary exists
2. Tests RPC server startup and shutdown
3. Tests distributed llama-server with RPC connections

Usage:
    python scripts/test_rpc_distributed.py [--model PATH]

Prerequisites:
    - llama.cpp built with GGML_RPC=ON
    - rpc-server and llama-server binaries in ~/llama.cpp/build/bin/
    - A GGUF model file for testing
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path


def print_header(message: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}\n")


def print_success(message: str) -> None:
    print(f"✓ {message}")


def print_error(message: str) -> None:
    print(f"✗ {message}")


def print_info(message: str) -> None:
    print(f"ℹ {message}")


def find_binary(name: str) -> Path | None:
    """Find a binary in common locations."""
    search_paths = [
        Path.home() / "llama.cpp" / "build" / "bin" / name,
        Path("/usr/local/bin") / name,
        Path("/usr/bin") / name,
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def is_port_available(port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def is_port_responding(port: int, host: str = "127.0.0.1") -> bool:
    """Check if something is responding on the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        try:
            sock.connect((host, port))
            return True
        except (OSError, socket.timeout):
            return False


def test_binaries() -> tuple[bool, Path | None, Path | None]:
    """Test that required binaries exist."""
    print_header("Testing Binary Availability")
    
    rpc_server = find_binary("rpc-server")
    llama_server = find_binary("llama-server")
    
    all_ok = True
    
    if rpc_server:
        print_success(f"rpc-server found: {rpc_server}")
    else:
        print_error("rpc-server not found")
        print_info("Build with: cd ~/llama.cpp && cmake -B build -DGGML_RPC=ON && cmake --build build --target rpc-server")
        all_ok = False
    
    if llama_server:
        print_success(f"llama-server found: {llama_server}")
    else:
        print_error("llama-server not found")
        print_info("Build with: cd ~/llama.cpp && cmake -B build && cmake --build build --target llama-server")
        all_ok = False
    
    return all_ok, rpc_server, llama_server


def test_rpc_server_lifecycle(rpc_server_path: Path) -> bool:
    """Test RPC server startup and shutdown."""
    print_header("Testing RPC Server Lifecycle")
    
    test_port = 50099
    
    if not is_port_available(test_port):
        print_error(f"Test port {test_port} not available")
        return False
    
    print_info(f"Starting rpc-server on port {test_port}...")
    
    process = subprocess.Popen(
        [str(rpc_server_path), "--host", "0.0.0.0", "--port", str(test_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    try:
        # Wait for server to start
        start_time = time.time()
        timeout = 10
        
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                stderr = process.stderr.read().decode() if process.stderr else ""
                print_error(f"rpc-server died: {stderr[:200]}")
                return False
            
            if is_port_responding(test_port):
                print_success(f"rpc-server responding on port {test_port}")
                break
            
            time.sleep(0.5)
        else:
            print_error(f"rpc-server failed to start within {timeout}s")
            process.kill()
            return False
        
        # Test graceful shutdown
        print_info("Testing graceful shutdown...")
        process.terminate()
        
        try:
            process.wait(timeout=5)
            print_success("rpc-server terminated gracefully")
        except subprocess.TimeoutExpired:
            print_error("rpc-server didn't terminate, killing...")
            process.kill()
            process.wait()
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Error: {e}")
        process.kill()
        return False


def test_distributed_inference(
    llama_server_path: Path,
    rpc_server_path: Path,
    model_path: str,
) -> bool:
    """Test distributed inference with one RPC worker."""
    print_header("Testing Distributed Inference")
    
    if not Path(model_path).exists():
        print_error(f"Model not found: {model_path}")
        return False
    
    rpc_port = 50098
    server_port = 8081
    
    print_info(f"Starting RPC worker on port {rpc_port}...")
    
    # Start RPC worker
    rpc_process = subprocess.Popen(
        [str(rpc_server_path), "--host", "0.0.0.0", "--port", str(rpc_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    try:
        # Wait for RPC server
        time.sleep(2)
        
        if not is_port_responding(rpc_port):
            print_error("RPC worker failed to start")
            rpc_process.kill()
            return False
        
        print_success("RPC worker started")
        
        # Start llama-server with RPC connection
        print_info(f"Starting llama-server with RPC connection...")
        
        server_cmd = [
            str(llama_server_path),
            "-m", model_path,
            "--port", str(server_port),
            "--host", "127.0.0.1",
            "-c", "512",
            "--rpc", f"127.0.0.1:{rpc_port}",
            "--tensor-split", "0.5,0.5",
        ]
        
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            # Wait for server to start
            start_time = time.time()
            timeout = 60  # Model loading can take a while
            
            while time.time() - start_time < timeout:
                if server_process.poll() is not None:
                    stderr = server_process.stderr.read().decode() if server_process.stderr else ""
                    print_error(f"llama-server died: {stderr[:500]}")
                    return False
                
                if is_port_responding(server_port):
                    print_success(f"llama-server responding on port {server_port}")
                    break
                
                time.sleep(1)
                print_info(f"Waiting for server... ({int(time.time() - start_time)}s)")
            else:
                print_error(f"llama-server failed to start within {timeout}s")
                server_process.kill()
                return False
            
            # Test inference
            print_info("Testing inference request...")
            
            import json
            import urllib.request
            
            request_data = json.dumps({
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "stream": False,
            }).encode("utf-8")
            
            request = urllib.request.Request(
                f"http://127.0.0.1:{server_port}/v1/chat/completions",
                data=request_data,
                headers={"Content-Type": "application/json"},
            )
            
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print_success(f"Inference successful: {content[:50]}...")
                    return True
            except Exception as e:
                print_error(f"Inference failed: {e}")
                return False
            
        finally:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
    
    finally:
        rpc_process.terminate()
        try:
            rpc_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rpc_process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test llama.cpp distributed RPC inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path.home() / ".exo" / "models" / "test.gguf"),
        help="Path to GGUF model file for testing",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the inference test (requires model file)",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  llama.cpp Distributed RPC Test Suite")
    print("="*60)
    
    # Test 1: Check binaries
    binaries_ok, rpc_server, llama_server = test_binaries()
    
    if not binaries_ok:
        print_header("FAILED: Required binaries not found")
        return 1
    
    assert rpc_server is not None
    assert llama_server is not None
    
    # Test 2: RPC server lifecycle
    if not test_rpc_server_lifecycle(rpc_server):
        print_header("FAILED: RPC server lifecycle test")
        return 1
    
    # Test 3: Distributed inference (optional)
    if not args.skip_inference:
        if not Path(args.model).exists():
            print_info(f"Model not found at {args.model}, skipping inference test")
            print_info("Run with --model PATH to test distributed inference")
        else:
            if not test_distributed_inference(llama_server, rpc_server, args.model):
                print_header("FAILED: Distributed inference test")
                return 1
    
    print_header("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

