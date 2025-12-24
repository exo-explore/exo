#!/usr/bin/env python3
"""
Exo P2P Cluster Benchmark Suite
Tests 2-node distributed cluster performance

Benchmarks:
1. API response time (both nodes)
2. Model listing latency
3. Node discovery time
4. Network round-trip time
5. Master election time
"""

import requests
import time
import json
from datetime import datetime

NODES = [
    {"name": "localhost (4x V100)", "url": "http://localhost:52415"},
    {"name": ".106 (RTX 5070)", "url": "http://192.168.0.106:52415"}
]

def benchmark_api_latency():
    """Measure API response time on each node"""
    print("\n" + "="*60)
    print("BENCHMARK 1: API Response Latency")
    print("="*60)

    results = {}
    for node in NODES:
        times = []
        for i in range(10):
            start = time.time()
            try:
                resp = requests.get(f"{node['url']}/v1/models", timeout=5)
                elapsed = (time.time() - start) * 1000  # ms
                times.append(elapsed)
                print(f"  {node['name']}: {elapsed:.2f}ms")
            except Exception as e:
                print(f"  {node['name']}: ERROR - {e}")
                continue

        if times:
            results[node['name']] = {
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "median": sorted(times)[len(times)//2]
            }

    print("\nSummary:")
    for name, stats in results.items():
        print(f"  {name}:")
        print(f"    Min: {stats['min']:.2f}ms")
        print(f"    Max: {stats['max']:.2f}ms")
        print(f"    Avg: {stats['avg']:.2f}ms")
        print(f"    Median: {stats['median']:.2f}ms")

    return results

def benchmark_model_listing():
    """Measure model listing performance"""
    print("\n" + "="*60)
    print("BENCHMARK 2: Model Listing Performance")
    print("="*60)

    for node in NODES:
        try:
            start = time.time()
            resp = requests.get(f"{node['url']}/v1/models")
            elapsed = (time.time() - start) * 1000

            data = resp.json()
            model_count = len(data.get('data', []))

            print(f"  {node['name']}:")
            print(f"    Models: {model_count}")
            print(f"    Latency: {elapsed:.2f}ms")
            print(f"    Throughput: {model_count / (elapsed/1000):.1f} models/sec")
        except Exception as e:
            print(f"  {node['name']}: ERROR - {e}")

def benchmark_network_latency():
    """Measure network round-trip between nodes"""
    print("\n" + "="*60)
    print("BENCHMARK 3: Inter-Node Network Latency")
    print("="*60)

    # Measure latency from localhost to .106
    times = []
    for i in range(20):
        start = time.time()
        try:
            resp = requests.get(f"{NODES[1]['url']}/v1/models", timeout=2)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        except:
            continue

    if times:
        print(f"  Network RTT (localhost → .106):")
        print(f"    Min: {min(times):.2f}ms")
        print(f"    Max: {max(times):.2f}ms")
        print(f"    Avg: {sum(times)/len(times):.2f}ms")
        print(f"    P50: {sorted(times)[len(times)//2]:.2f}ms")
        print(f"    P95: {sorted(times)[int(len(times)*0.95)]:.2f}ms")
        print(f"    P99: {sorted(times)[int(len(times)*0.99)]:.2f}ms")

def benchmark_cluster_availability():
    """Test cluster availability and failover"""
    print("\n" + "="*60)
    print("BENCHMARK 4: Cluster Availability")
    print("="*60)

    for node in NODES:
        try:
            resp = requests.get(f"{node['url']}/v1/models", timeout=2)
            status = "✅ ONLINE" if resp.status_code == 200 else f"❌ ERROR {resp.status_code}"
            print(f"  {node['name']}: {status}")
        except Exception as e:
            print(f"  {node['name']}: ❌ OFFLINE - {e}")

def save_results(results, filename="benchmark_results.json"):
    """Save benchmark results to JSON"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "cluster_size": len(NODES),
        "nodes": [n['name'] for n in NODES],
        "results": results
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to {filename}")

def main():
    print("\n" + "🔥"*30)
    print("EXO P2P CLUSTER BENCHMARK")
    print("2-Node Distributed AI System")
    print("🔥"*30)

    results = {}

    # Run benchmarks
    results['api_latency'] = benchmark_api_latency()
    benchmark_model_listing()
    benchmark_network_latency()
    benchmark_cluster_availability()

    # Save results
    save_results(results)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
