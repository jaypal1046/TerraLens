import ctypes
import numpy as np
import os

class FastRadar:
    def __init__(self, dll_path):
        self.lib = ctypes.CDLL(dll_path)
        
        # Define function signatures
        self.lib.compute_loss_fast.restype = ctypes.c_double
        self.lib.compute_loss_fast.argtypes = [
            ctypes.c_double, ctypes.c_double, 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
            ctypes.c_int
        ]
        
        self.lib.probe_fast.argtypes = [
            ctypes.c_double, ctypes.c_double, 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
            ctypes.c_int, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
        ]

    def probe(self, w1, w2, x, y, h=1e-5):
        n = len(x)
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        grad = np.zeros(2, dtype=np.float64)
        curv = np.zeros(2, dtype=np.float64)
        
        grad_ptr = grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        curv_ptr = curv.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.probe_fast(w1, w2, x_ptr, y_ptr, n, h, grad_ptr, curv_ptr)
        return grad, curv

# --- PHASE 2: EVOLUTIONARY MAPPING ---

class EvolutionaryMap:
    def __init__(self):
        self.history = [] # Stores (neuron_count, landscape_grid)

    def record_state(self, neuron_count, w1_range, w2_range, loss_grid):
        """Records the landscape for a specific architecture size"""
        # Find minimum of the RECORDED state
        min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
        min_coords = (w1_range[min_idx[1]], w2_range[min_idx[0]])
        
        state = {
            'neurons': neuron_count,
            'grid': loss_grid.copy(),
            'w1': w1_range,
            'w2': w2_range,
            'min_coords': min_coords
        }
        self.history.append(state)
        print(f"MAP: Recorded landscape for {neuron_count} neurons. Min at {min_coords}")

    def predict_next_minimum(self):
        """
        PREDICTS where the minimum will be at the NEXT neuron addition.
        Uses linear extrapolation of the last two known minimum locations.
        """
        if len(self.history) < 1:
            return (0.0, 0.0) # Default
        
        if len(self.history) == 1:
            return self.history[0]['min_coords'] # Stay at current
        
        # Extrapolate: m2 + (m2 - m1)
        m1 = np.array(self.history[-2]['min_coords'])
        m2 = np.array(self.history[-1]['min_coords'])
        
        prediction = m2 + (m2 - m1)
        # Clip to common bounds
        prediction = np.clip(prediction, -5, 5)
        
        print(f"MAP PREDICTION: Next minimum expected near {prediction}")
        return tuple(prediction)

    def calculate_delta(self):
        """Calculates HOW the landscape shifted since the last neuron was added"""
        if len(self.history) < 2:
            return None
        
        s1 = self.history[-2]['grid']
        s2 = self.history[-1]['grid']
        
        delta = s2 - s1
        return delta
