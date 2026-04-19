import numpy as np

class SkipEngine:
    def __init__(self, base_jump_factor=1.0):
        self.base_jump_factor = base_jump_factor

    def calculate_skip(self, weights, gradient, curvature):
        """
        Calculates the next position by jumping over negative curvature (mountains).
        Logic: Distance is inversely proportional to curvature magnitude.
        """
        w = np.array(weights)
        g = np.array(gradient)
        c = np.array(curvature)
        
        # We only skip in directions where curvature is negative (mountains)
        # For directions with positive curvature, we stay (skip_val = 0)
        skip_dirs = np.where(c < 0, 1.0, 0.0)
        
        # Calculate jump distance: Shallow mountain (small |c|) -> Large jump
        # We use 1 / (abs(c) + epsilon)
        epsilon = 1e-6
        jump_magnitudes = self.base_jump_factor / (np.abs(c) + epsilon)
        
        # Jump direction: Move AWAY from the gradient (climb down) or 
        # simply jump further in the direction of current momentum?
        # Let's jump in the direction of the negative gradient but with the skip magnitude.
        jump_vector = -np.sign(g) * jump_magnitudes * skip_dirs
        
        new_weights = w + jump_vector
        return new_weights

    def validate_skip(self, old_loss, new_loss):
        """If skip result is worse, we might need to adjust strategy"""
        return new_loss < old_loss
