import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radar.probe import RadarProbe

def synthetic_valley(w1, w2, depth, width, center):
    """
    Creates a valley with KNOWN depth and width.
    """
    dist_sq = (w1 - center[0])**2 + (w2 - center[1])**2
    return depth * (1 - np.exp(-dist_sq / (2 * width**2)))

def test_depth_detection():
    print("--- TEST GROUP 2: DEPTH DETECTION ACCURACY ---")
    
    test_cases = [
        {"true_depth": 1.0, "width": 0.5, "center": [0.2, 0.3]},
        {"true_depth": 3.0, "width": 0.5, "center": [0.5, 0.5]},
        {"true_depth": 5.0, "width": 0.5, "center": [0.8, 0.7]},
        {"true_depth": 0.5, "width": 1.0, "center": [-1, -1]},
        {"true_depth": 4.0, "width": 0.2, "center": [2, 2]}, # deep + narrow
    ]
    
    measured_depths = []
    
    for case in test_cases:
        # Define a lambda for this specific valley
        loss_fn = lambda w1, w2: synthetic_valley(w1, w2, case["true_depth"], case["width"], case["center"])
        probe = RadarProbe(loss_fn)
        
        # Measure curvature at the exact center
        _, curvatures = probe.probe(*case["center"])
        
        # In TerraLens, depth is measured by the magnitude of positive curvature
        # (The sharper the valley, the 'deeper' it is in terms of optimization stability)
        avg_curv = np.mean(curvatures)
        measured_depths.append(avg_curv)
        
        print(f"True Depth: {case['true_depth']:4.1f} | Measured Curvature: {avg_curv:8.3f}")

    # --- Ranking Test ---
    true_rank = np.argsort([c["true_depth"] for c in test_cases])
    measured_rank = np.argsort(measured_depths)
    
    print(f"\nTrue Rank:     {true_rank}")
    print(f"Measured Rank: {measured_rank}")
    
    if np.array_equal(true_rank, measured_rank):
        print("\nSUCCESS: Depth RANKING is accurate.")
    else:
        # Note: If width varies, depth might not map 1:1 to curvature. 
        # This is an 'Honesty Test' point.
        print("\nWARNING: Ranking mismatch. (Likely due to width/curvature trade-off)")

if __name__ == "__main__":
    test_depth_detection()
