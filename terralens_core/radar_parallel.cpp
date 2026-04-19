#include <cmath>
#include <vector>
#include <thread>
#include <mutex>

typedef double (*LossFn)(double*, int);

extern "C" {
    // Worker function for threading
    void radar_worker(double* weights, int start, int end, int total_n, double h, 
                      LossFn loss_fn, double* out_curvatures) {
        for (int i = start; i < end; i++) {
            double original_val = weights[i];
            
            weights[i] = original_val + h;
            double l_plus = loss_fn(weights, total_n);
            
            weights[i] = original_val - h;
            double l_minus = loss_fn(weights, total_n);
            
            weights[i] = original_val;
            double l_center = loss_fn(weights, total_n);
            
            out_curvatures[i] = (l_plus - 2 * l_center + l_minus) / (h * h);
        }
    }

    // High-level Parallel Block Scan
    void radar_parallel_scan(double* weights, int n, double h, 
                             LossFn loss_fn, double* out_curvatures, int num_threads) {
        
        std::vector<std::thread> threads;
        int chunk_size = n / num_threads;
        
        for (int i = 0; i < num_threads; i++) {
            int start = i * chunk_size;
            int end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
            threads.emplace_back(radar_worker, weights, start, end, n, h, loss_fn, out_curvatures);
        }

        for (auto& t : threads) {
            t.join();
        }
    }
}
