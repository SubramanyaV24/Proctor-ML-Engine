#!/usr/bin/env python3
"""
Quick benchmark to measure /run_all endpoint latency with a real frame.
"""
import base64
import time
import requests
import sys

def create_test_image():
    """Create a minimal 640x480 JPEG test image."""
    import cv2
    import numpy as np
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def benchmark_run_all(url="http://localhost:8000/run_all", iterations=10):
    image_b64 = create_test_image()
    
    latencies = []
    for i in range(iterations):
        payload = {
            "image_base64": image_b64,
            "session_id": "benchmark"
        }
        
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=30)
        elapsed = time.perf_counter() - t0
        
        if resp.status_code == 200:
            latencies.append(elapsed * 1000)  # convert to ms
            result = resp.json()
            print(f"[{i+1}/{iterations}] {elapsed*1000:.1f}ms | "
                  f"face={result['analysis'].get('faceDetected')} | "
                  f"processing={result['analysis'].get('processingTimeMs')}ms")
        else:
            print(f"[{i+1}/{iterations}] ERROR: {resp.status_code}")
    
    if latencies:
        print(f"\n{'='*60}")
        print(f"Results over {len(latencies)} requests:")
        print(f"  Min:    {min(latencies):.1f}ms")
        print(f"  Median: {sorted(latencies)[len(latencies)//2]:.1f}ms")
        print(f"  Max:    {max(latencies):.1f}ms")
        print(f"  Avg:    {sum(latencies)/len(latencies):.1f}ms")
        print(f"{'='*60}")
        
        # Estimate concurrent capacity
        avg_ms = sum(latencies) / len(latencies)
        throughput = 1000 / avg_ms
        print(f"\nEstimated capacity:")
        print(f"  ~{throughput:.1f} requests/second (single-threaded)")
        print(f"  ~{throughput * 6:.0f} requests/second (6 cores, ideal parallelism)")
        print(f"  Comfortable concurrent users (1 req/sec): {int(throughput * 6 * 0.7)}")

if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Benchmarking /run_all with {iterations} iterations...\n")
    benchmark_run_all(iterations=iterations)
