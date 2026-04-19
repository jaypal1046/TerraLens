#include <cmath>
#include <vector>

extern "C" {
    // Sigmoid activation
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // High-speed MSE Loss calculation for the 2-weight model
    double compute_loss_fast(double w1, double w2, double* x, double* y, int n) {
        double total_loss = 0.0;
        for (int i = 0; i < n; i++) {
            double pred = sigmoid(w1 * x[i] + w2);
            double diff = pred - y[i];
            total_loss += diff * diff;
        }
        return total_loss / n;
    }

    // Fast Radar Probe: Gradient and Curvature (Hessian Diagonal)
    void probe_fast(double w1, double w2, double* x, double* y, int n, double h, 
                    double* out_grad, double* out_curv) {
        
        // Gradient (Central Difference)
        double loss_w1_plus = compute_loss_fast(w1 + h, w2, x, y, n);
        double loss_w1_minus = compute_loss_fast(w1 - h, w2, x, y, n);
        out_grad[0] = (loss_w1_plus - loss_w1_minus) / (2 * h);

        double loss_w2_plus = compute_loss_fast(w1, w2 + h, x, y, n);
        double loss_w2_minus = compute_loss_fast(w1, w2 - h, x, y, n);
        out_grad[1] = (loss_w2_plus - loss_w2_minus) / (2 * h);

        // Curvature (Hessian Diagonal)
        double loss_center = compute_loss_fast(w1, w2, x, y, n);
        out_curv[0] = (loss_w1_plus - 2 * loss_center + loss_w1_minus) / (h * h);
        out_curv[1] = (loss_w2_plus - 2 * loss_center + loss_w2_minus) / (h * h);
    }
}
