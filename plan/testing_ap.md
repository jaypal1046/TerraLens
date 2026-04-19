TerraLens evaluation and testing framework
Evaluation framework

3 levels of proof — correctness, comparison, real-world impact

Level 1 — Correctness (known answer tests)
Do first
Use functions where the TRUE global minimum is mathematically known. Your system must find it.

Rosenbrock function — known minimum at (1,1)
Classic test. Very hard — deep narrow curved valley. If skip engine works, it finds (1,1).
Rastrigin function — known minimum at (0,0,...,0)
Many local minima. Perfect test for skip engine. Standard benchmark every optimizer is tested on.
Ackley function — known minimum at (0,0)
Nearly flat outer region with deep central minimum. Tests satellite scanner's ability to find it.
Beale, Himmelblau, Booth functions
All have known exact answers. Run all of them. Pass = system is mathematically correct.
from scipy.optimize import rosen
import numpy as np

def rosenbrock(w):
    return rosen(w)

true_minimum = np.array([1.0, 1.0])
result = terralens.optimize(rosenbrock, start=np.array([0.0, 0.0]))

error = np.linalg.norm(result - true_minimum)
assert error < 0.01, f"FAILED: error={error:.4f}"
print(f"PASSED: found minimum at {result}, error={error:.6f}")
Pass criteria

Distance to true minimum

Error < 0.01

Success rate across 100 random starts

> 90% find correct answer

Level 2 — Comparison (beat the baselines)
Most important
Run TerraLens AND the standard optimizers on identical problems. Measure everything.

Baseline 1 — SGD (standard gradient descent)
The oldest method. Easiest to beat. You should clearly outperform this.
Baseline 2 — Adam optimizer
Current industry standard. This is your real target. If you match Adam, good. If you beat Adam, publishable.
Baseline 3 — Simulated Annealing
Also uses random jumping. Shows your guided skip is better than random jumping.
Baseline 4 — Basin Hopping
Most similar existing method to your skip engine. Must clearly outperform this.
What to measure for each optimizer

Steps to convergence

How many gradient evaluations to reach 95% accuracy?

Final loss value

Which optimizer finds the lowest loss overall?

Wall clock time

Actual seconds. Includes all TerraLens overhead.

Variance across runs

Run 10 times each. How consistent are results?

results = {}
optimizers = {
    "SGD":        torch.optim.SGD(model.parameters(), lr=0.01),
    "Adam":       torch.optim.Adam(model.parameters(), lr=1e-3),
    "TerraLens":  TerraLensOptimizer(model.parameters(), lr=1e-3),
}

for name, opt in optimizers.items():
    acc, steps, time = train_and_measure(model, opt, dataset)
    results[name] = {"accuracy": acc, "steps": steps, "time": time}

print(pd.DataFrame(results).T)
Level 3 — Ablation (prove WHICH part helps)
For the paper
Turn off each component one at a time. Measure what happens. This proves each piece contributes.

TerraLens full system
All 4 layers active. This is your best result.
TerraLens — no satellite (remove Layer 1)
Does global scanning actually help? Difference proves it.
TerraLens — no skip engine (remove Layer 4)
This is the key test. Does the skip mechanism actually escape local minima? Most important ablation.
TerraLens — no map-while-building
Does tracking landscape during construction actually help training? Proves your novel idea.
configs = {
    "Full TerraLens":          TerraLensOptimizer(use_satellite=True,  skip=True,  morph=True),
    "No satellite":            TerraLensOptimizer(use_satellite=False, skip=True,  morph=True),
    "No skip engine":          TerraLensOptimizer(use_satellite=True,  skip=False, morph=True),
    "No map-while-building":   TerraLensOptimizer(use_satellite=True,  skip=True,  morph=False),
}
for name, opt in configs.items():
    print(name, "→", train_and_measure(model, opt, dataset))
If removing a component makes no difference → that component is not working. That is critical feedback, not failure. Fix it.

Level 4 — Honesty tests (where does it fail?)
Builds trust
Every real research paper shows where the method fails. This builds credibility. Find the limits before reviewers do.

Test on convex problems — TerraLens should NOT help here
Convex = only one minimum. Adam already solves this perfectly. If TerraLens is slower here, that is expected and honest.
Test with very small networks — overhead may hurt
TerraLens adds compute. For tiny networks the overhead may cost more than the skip saves. Show this honestly.
Test at massive scale — where does it break?
Find the network size where TerraLens stops being faster. That becomes the "future work" section of your paper.
Test with bad hyperparameters — how sensitive is it?
If skip_factor=2.0 works but skip_factor=1.9 fails → system is fragile. Good systems work across a range of settings.
Your current result vs what is needed

96.6% MNIST accuracy

Good starting point

vs Adam on same data?

Not yet measured

Rastrigin / Rosenbrock tests?

Not yet run

Ablation study done?

Not yet done

Write the full test suite ↗ Write ablation study ↗