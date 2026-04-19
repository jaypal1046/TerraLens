import numpy as np

class SatelliteScanner:
    def __init__(self, bounds, grid_size=(10, 10)):
        """
        bounds: list of (min, max) for each dimension
        grid_size: number of divisions per dimension
        """
        self.bounds = bounds
        self.grid_size = grid_size

    def scan(self, loss_fn):
        """
        Performs a global sparse scan.
        Returns the top 20% 'candidate' regions.
        """
        results = []
        
        # Generate grid midpoints for sampling
        w1_steps = np.linspace(self.bounds[0][0], self.bounds[0][1], self.grid_size[0] + 1)
        w2_steps = np.linspace(self.bounds[1][0], self.bounds[1][1], self.grid_size[1] + 1)
        
        for i in range(len(w1_steps)-1):
            for j in range(len(w2_steps)-1):
                # Sample random point in this box
                w1 = np.random.uniform(w1_steps[i], w1_steps[i+1])
                w2 = np.random.uniform(w2_steps[j], w2_steps[j+1])
                
                loss = loss_fn(w1, w2)
                results.append({
                    'coords': (w1, w2),
                    'bounds': ((w1_steps[i], w1_steps[i+1]), (w2_steps[j], w2_steps[j+1])),
                    'loss': loss
                })
        
        # Sort by loss and take bottom 20% (lowest loss)
        results.sort(key=lambda x: x['loss'])
        threshold_idx = max(1, int(len(results) * 0.2))
        candidates = results[:threshold_idx]
        
        return candidates
