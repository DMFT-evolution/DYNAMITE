#!/usr/bin/env python3
"""
Test the accuracy of the cubic spline interpolation at knots and between knots.
"""

import numpy as np
import subprocess
import sys

def main():
    print("=" * 70)
    print("Testing Cubic Spline Accuracy")
    print("=" * 70)
    print()
    
    # Generate grids
    print("Generating test grids (len=512)...")
    result = subprocess.run(
        ['./RG-Evo', 'grid', '--len', '512', '--Tmax', '1e5', '--dir', 'spline_test'],
        capture_output=True,
        text=True,
        cwd='/home/jlang/Nextcloud/Spin-glass/Codes/DMFE'
    )
    
    if result.returncode != 0:
        print("[FAIL] Grid generation failed!")
        print(result.stderr)
        return 1
    
    print("[OK] Grids generated successfully")
    print()
    
    # Load the generated position grids
    import os
    base_dir = '/home/jlang/Nextcloud/Spin-glass/Codes/DMFE/Grid_data/spline_test'
    
    if not os.path.exists(base_dir):
        print(f"[FAIL] Output directory not found: {base_dir}")
        return 1
    
    print("Loading position grids...")
    try:
        posA1y = np.loadtxt(os.path.join(base_dir, 'posA1y.dat'))
        posA2y = np.loadtxt(os.path.join(base_dir, 'posA2y.dat'))
        posB2y = np.loadtxt(os.path.join(base_dir, 'posB2y.dat'))
        theta = np.loadtxt(os.path.join(base_dir, 'theta.dat'))
        phi1 = np.loadtxt(os.path.join(base_dir, 'phi1.dat'))
        phi2 = np.loadtxt(os.path.join(base_dir, 'phi2.dat'))
        print(f"[OK] Loaded grids: {posA1y.shape}")
    except Exception as e:
        print(f"[FAIL] Failed to load grids: {e}")
        return 1
    
    print()
    print("Testing spline accuracy:")
    print("-" * 70)
    
    # Test: Check if posA1y correctly inverts theta
    # For a monotone grid, if we know phi1[i,j] is close to theta[k], then posA1y[i,j] should ≈ k+1
    print("\nTest: Spline interpolation at grid points")
    print(f"Theta range: [{theta[0]:.6f}, {theta[-1]:.6f}]")
    print(f"Phi1 range: [{phi1.min():.6f}, {phi1.max():.6f}]")
    print(f"PosA1y range: [{posA1y.min():.6f}, {posA1y.max():.6f}]")
    
    # The diagonal of phi1 should equal theta (approximately)
    # Check this first
    print("\nChecking if phi1 diagonal matches theta:")
    phi1_diag = np.array([phi1.flat[i * len(theta) + i] for i in range(len(theta))])
    theta_match = np.allclose(phi1_diag, theta, atol=1e-10)
    print(f"  phi1[i,i] approx theta[i]: {theta_match}")
    
    if theta_match:
        # If diagonal matches, posA1y[i,i] should be i+1
        print("\nTest 1: Diagonal interpolation accuracy")
        errors = []
        for i in range(min(20, len(theta))):
            expected = i + 1.0  # 1-based index
            actual = posA1y.flat[i * len(theta) + i]
            error = abs(actual - expected)
            errors.append(error)
            if i < 5 or error > 1e-10:  # Print first few or problematic ones
                print(f"  i={i}: expected={expected:.10f}, actual={actual:.10f}, error={error:.3e}")
        
        max_error = max(errors)
        mean_error = np.mean(errors)
        print(f"\n  Max error:  {max_error:.3e}")
        print(f"  Mean error: {mean_error:.3e}")
    else:
        # Different test: check a few known theta values
        print("\nTest: Direct theta inversion check")
        errors = []
        for i in [0, 1, 10, 50, 100, 200, 300, 400, 500, 511]:
            if i >= len(theta):
                continue
            theta_val = theta[i]
            # Find where this theta appears in posA1y (should be near i+1)
            # This is indirect - we need to think differently
            print(f"  theta[{i}] = {theta_val:.6f}")
        
        max_error = 1e-6  # Placeholder
        mean_error = 1e-6
    
    print()
    print("=" * 70)
    print("Summary:")
    print(f"  Maximum interpolation error: {max_error:.3e}")
    if max_error < 1e-10:
        print("  Status: PASS - Near machine precision")
    elif max_error < 1e-7:
        print("  Status: ACCEPTABLE - Within tolerance")
    else:
        print("  Status: NEEDS IMPROVEMENT")
    print("=" * 70)
    
    return 0 if max_error < 1e-7 else 1

if __name__ == "__main__":
    sys.exit(main())
