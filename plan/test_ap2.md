TerraLens neuron map and depth detection test designs
Neuron map + depth detection tests

These are novel tests — they do not exist anywhere else. You have to build them yourself.

Test Group 1 — Neuron map correctness
Map while building
Does your map accurately record how landscape changes as neurons are added?

The core problem to solve first

When you add neuron 3, your map says "a new valley appeared at position X." How do you VERIFY that is true? You need a ground truth to compare against.

1
Build a controlled test network
Use only 2 weights total. Then you can compute the FULL landscape by brute force (check every possible w1, w2 combination). This is your ground truth.
2
Add neurons one at a time, recompute full landscape each time
0 neurons → full brute force landscape → save it. Add neuron 1 → full brute force landscape → save it. Add neuron 2 → full brute force landscape → save it.
3
Run your map tracker alongside — compare its output to ground truth
Your map says "valley moved from (0.3, 0.5) to (0.4, 0.6) when neuron 2 was added." Ground truth says the same? PASS. Different? Your tracker has a bug.
def test_map_correctness():
    network = TinyNetwork(n_weights=2)
    tracker = LandscapeTracker()
    
    for n_neurons in range(1, 6):
        network.add_neuron()
        
        # Ground truth: brute force scan ALL (w1,w2) combinations
        ground_truth = brute_force_landscape(network, resolution=100)
        true_valleys = find_valleys(ground_truth)
        true_minimum = find_global_min(ground_truth)
        
        # What your map recorded
        map_valleys   = tracker.get_valleys(step=n_neurons)
        map_minimum   = tracker.get_predicted_minimum(step=n_neurons)
        
        # Compare
        valley_error  = compare_valley_locations(true_valleys, map_valleys)
        minimum_error = distance(true_minimum, map_minimum)
        
        print(f"Neuron {n_neurons}: valley_error={valley_error:.4f}, min_error={minimum_error:.4f}")
        assert minimum_error < 0.05, f"Map is WRONG at neuron {n_neurons}"
What to measure at each neuron addition

Valley location accuracy

Map predicted valley at X. Truth is at Y. Distance between them.

New valley detection

Did adding this neuron create a new valley? Did map detect it?

Destroyed valley detection

Did adding this neuron remove an old valley? Did map catch it?

Minimum shift tracking

Global minimum moved when neuron added. How accurately did map track the shift?

The visual test — most powerful proof

Plot both landscapes side by side at each neuron step
Left = ground truth (brute force). Right = your map's prediction. They should look identical. If they don't, you see exactly where the error is.
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for step in range(5):
    axes[0, step].contourf(ground_truth[step])
    axes[0, step].set_title(f"Ground truth — neuron {step+1}")
    axes[1, step].contourf(map_prediction[step])
    axes[1, step].set_title(f"Map prediction — neuron {step+1}")
plt.savefig("map_accuracy_visual.png")
Test Group 2 — Depth detection accuracy
Radar probe
When radar says "this valley has depth D" — is D actually correct?

Depth is the hardest thing to measure because unlike a real valley you cannot drop a rope in. You need a mathematical definition of depth first.

What does "depth" mean mathematically?

Definition 1 — Loss depth
Depth = (loss at valley rim) minus (loss at valley bottom). Bigger difference = deeper valley.
Definition 2 — Curvature depth
Depth = eigenvalue magnitude of Hessian at bottom. Large eigenvalue = sharp deep valley. Small = shallow flat valley.
Definition 3 — Basin width
How wide is the valley? Wide + deep = robust minimum. Narrow + deep = sharp minimum (bad for generalization).
1
Create synthetic valleys with known depth
Build artificial loss functions where YOU control the depth. Then test if radar measures it correctly.
def synthetic_valley(depth, width, center):
    """
    Creates a valley with KNOWN depth and width.
    depth = how deep the valley is (ground truth)
    width = how wide the valley is (ground truth)
    """
    def loss(w):
        distance = np.linalg.norm(w - center)
        return depth * (1 - np.exp(-distance**2 / (2 * width**2)))
    return loss

# Create 5 valleys with known depths
test_cases = [
    synthetic_valley(depth=1.0, width=0.5, center=[0.2, 0.3]),
    synthetic_valley(depth=2.0, width=0.5, center=[0.5, 0.5]),
    synthetic_valley(depth=5.0, width=0.5, center=[0.8, 0.2]),
    synthetic_valley(depth=0.5, width=2.0, center=[0.1, 0.9]),  # shallow+wide
    synthetic_valley(depth=5.0, width=0.1, center=[0.6, 0.7]),  # deep+narrow
]

for true_depth, loss_fn in zip([1.0, 2.0, 5.0, 0.5, 5.0], test_cases):
    measured_depth = terralens.radar.measure_depth(loss_fn, position=valley_center)
    error = abs(measured_depth - true_depth) / true_depth * 100
    print(f"True depth: {true_depth:.1f} | Measured: {measured_depth:.3f} | Error: {error:.1f}%")
2
Test if deeper valleys are ranked correctly
You don't need exact depth numbers. You need relative ranking: valley A deeper than valley B. Radar must agree with ground truth ranking.
def test_depth_ranking():
    valleys = [
        {"center": [0.2, 0.3], "true_depth": 1.0},
        {"center": [0.5, 0.5], "true_depth": 3.0},
        {"center": [0.8, 0.7], "true_depth": 5.0},
    ]
    
    measured = [terralens.radar.measure_depth(v["center"]) for v in valleys]
    true_rank = sorted(range(3), key=lambda i: valleys[i]["true_depth"])
    measured_rank = sorted(range(3), key=lambda i: measured[i])
    
    assert true_rank == measured_rank, "Depth RANKING is wrong — radar cannot order valleys"
    print("Ranking test PASSED")
3
Test the critical question — does deeper mean better training?
This is the most important test. If radar finds "deepest valley" and optimizer starts there, does it train better? If yes, depth detection is truly useful. If no, you need to rethink what depth means.
def test_depth_predicts_training_quality():
    # Find top 3 valleys by TerraLens depth measurement
    valleys = terralens.satellite.scan(loss_landscape)
    valleys_sorted = sorted(valleys, key=lambda v: v.depth, reverse=True)
    
    results = []
    for valley in valleys_sorted[:3]:
        # Start training FROM each valley
        model = reset_model(init_weights=valley.center)
        final_loss = train_to_convergence(model)
        results.append({"depth": valley.depth, "final_loss": final_loss})
    
    # Deeper valley should give lower final loss
    correlation = pearsonr([r["depth"] for r in results],
                           [-r["final_loss"] for r in results])
    print(f"Depth vs training quality correlation: {correlation:.3f}")
    assert correlation > 0.7, "Depth metric does not predict training quality"
Pass criteria for depth detection

Depth measurement error

< 15% on synthetic valleys

Ranking accuracy

> 85% correct ordering

Depth vs quality correlation

> 0.7 Pearson score

Deep + wide vs deep + narrow

System must distinguish these (width matters too)

Test Group 3 — Combined: map guides depth search
The full system test
Does building the map during construction help you find deeper valleys faster during training?

This is the test that proves your whole original idea. Map while building → know where deep valleys are before training starts → train faster.

Experiment A — cold start (no map)
Standard random initialization. Train from scratch. Measure steps to reach target loss.
Experiment B — warm start (using map)
Map built during construction. Start training at position map predicted as deepest valley. Measure steps to reach same target loss.
Difference = the value of your map
If B reaches target in 30% fewer steps than A, your map-while-building idea is proven. That number goes in the paper title.
def test_map_guided_initialization():
    results = {"cold_start": [], "warm_start": []}
    
    for trial in range(20):  # run 20 times for statistical significance
        
        # Experiment A: cold start
        model_cold = build_network(n_neurons=10)
        steps_cold = train_until_convergence(model_cold, target_loss=0.1)
        results["cold_start"].append(steps_cold)
        
        # Experiment B: warm start using map
        model_warm = build_network_with_tracker(n_neurons=10)
        best_init  = model_warm.terralens_map.get_deepest_valley()
        model_warm.set_weights(best_init)
        steps_warm = train_until_convergence(model_warm, target_loss=0.1)
        results["warm_start"].append(steps_warm)
    
    avg_cold = np.mean(results["cold_start"])
    avg_warm = np.mean(results["warm_start"])
    improvement = (avg_cold - avg_warm) / avg_cold * 100
    
    print(f"Cold start avg steps: {avg_cold:.0f}")
    print(f"Warm start avg steps: {avg_warm:.0f}")
    print(f"Improvement: {improvement:.1f}%")
    
    assert improvement > 0, "Map initialization provides no benefit"
Run these in order

First

Test Group 1 — verify map records changes correctly

Second

Test Group 2 — verify depth measurement is accurate

Third

Test Group 3 — verify map + depth together improve training

If Group 3 passes

You have proven the core novel contribution of TerraLens

Write brute force landscape code ↗ Write depth test code ↗