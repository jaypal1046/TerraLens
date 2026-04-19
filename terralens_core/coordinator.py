import ctypes
import numpy as np
import os

class BlockCoordinator:
    def __init__(self, dll_path, block_size=1000):
        self.lib = ctypes.CDLL(dll_path)
        self.block_size = block_size
        
        # Define function signature
        # void radar_parallel_scan(double* weights, int n, double h, LossFn loss_fn, double* out_curvatures, int num_threads)
        self.lib.radar_parallel_scan.argtypes = [
            ctypes.POINTER(ctypes.c_double), 
            ctypes.c_int, 
            ctypes.c_double, 
            ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]

    def coordinate_scan(self, full_weights, loss_callback, num_threads=4):
        """
        Scans a massive weight array by breaking it into blocks.
        """
        n = len(full_weights)
        curvatures = np.zeros(n, dtype=np.float64)
        
        for start in range(0, n, self.block_size):
            end = min(start + self.block_size, n)
            block = full_weights[start:end].astype(np.float64)
            
            # Prepare pointers
            block_ptr = block.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            curv_ptr = curvatures[start:end].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            # Call Parallel Radar
            self.lib.radar_parallel_scan(
                block_ptr, (end - start), 1e-5, 
                loss_callback, curv_ptr, num_threads
            )
            
        return curvatures

    def get_skip_indices(self, curvatures, threshold=-0.01):
        """Identifies indices that are currently in 'Mountain' regions"""
        return np.where(curvatures < threshold)[0]
