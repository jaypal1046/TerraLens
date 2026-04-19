import numpy as np

class RadarProbe:
    def __init__(self, loss_fn, epsilon=1e-5):
        self.loss_fn = loss_fn
        self.epsilon = epsilon

    def probe(self, w1, w2):
        """
        Measures gradient and curvature at specific coordinates.
        Returns: (grad_w1, grad_w2), (curv_w1, curv_w2)
        """
        h = self.epsilon
        
        # Central difference for gradient
        g1 = (self.loss_fn(w1 + h, w2) - self.loss_fn(w1 - h, w2)) / (2 * h)
        g2 = (self.loss_fn(w1, w2 + h) - self.loss_fn(w1, w2 - h)) / (2 * h)
        
        # Second derivative for Hessian diagonal (Curvature)
        # f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2
        center_loss = self.loss_fn(w1, w2)
        c1 = (self.loss_fn(w1 + h, w2) - 2*center_loss + self.loss_fn(w1 - h, w2)) / (h**2)
        c2 = (self.loss_fn(w1, w2 + h) - 2*center_loss + self.loss_fn(w1, w2 - h)) / (h**2)
        
        return (g1, g2), (c1, c2)
