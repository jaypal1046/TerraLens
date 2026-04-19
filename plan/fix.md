TerraLens test audit — critical issues and fixes
Test audit report

What the tests prove, what they don't, and what must be fixed before publishing

Critical flaw — Test Group 3 is not a fair test
Must fix
Look at this code from your test_group3_combined.py:

best_init = torch.tensor([2.0, 2.0])  # <-- YOU HARDCODED THE ANSWER
You manually put in (2.0, 2.0) which is exactly where the deepest valley is in your landscape function. The warm start is not using TerraLens to FIND the best point — you told it the answer in advance. Of course it performs better. Anyone starting near the correct answer will get a lower loss.

What this test actually proves
Starting near the true minimum gives lower loss than starting randomly. This is trivially true and proves nothing about TerraLens.
What it needs to prove
That TerraLens can FIND (2.0, 2.0) on its own, without you telling it, and that this ability leads to better training.
The fix — TerraLens must find the point itself

def test_map_guided_initialization_FIXED():
    landscape = build_complex_landscape()  # has valleys at unknown locations
    
    # Warm start: TerraLens scans and FINDS the best point
    scanner = SatelliteScanner(landscape)
    candidates = scanner.scan()             # TerraLens does the work
    best_init = candidates[0]["coords"]     # TerraLens found this, not you
    
    # NOW compare cold vs warm
    # If best_init is near (2,2) → TerraLens worked
    # If best_init is wrong → TerraLens failed honestly
    print(f"TerraLens found: {best_init}")
    print(f"True answer:     [2.0, 2.0]")
    
    # Then run training comparison
    cold_loss = train_from_random(landscape, n_trials=50)
    warm_loss  = train_from(best_init, landscape, n_trials=50)
    improvement = (cold_loss - warm_loss) / cold_loss * 100
    print(f"Honest improvement: {improvement:.1f}%")
Important issue — Test Group 1 map is not predicting
Needs fix
Look at this in test_group1_map_correctness.py:

ev_map.record_state(n, w1_range, w2_range, truth_grid)
map_min_idx = np.unravel_index(np.argmin(truth_grid), truth_grid.shape)
# ^ using truth_grid again, not the map's prediction!
The map is recording the ground truth grid and then finding the minimum OF the ground truth grid. It is not making its own prediction at all. Zero error is guaranteed — you compared truth to itself.

What the map must actually do
At neuron N, predict where the minimum will be at neuron N+1 BEFORE adding neuron N+1. Then verify by comparing to ground truth after.
ev_map.record_state(n, truth_grid)

# Predict NEXT state before adding next neuron
predicted_min = ev_map.predict_next_minimum()

# Now add neuron and compute ground truth
loss_fn_next = get_loss_fn(n + 1)
_, _, next_grid = brute_force_scan(loss_fn_next)
true_next_min  = find_minimum(next_grid)

# NOW compare prediction to truth
error = distance(predicted_min, true_next_min)
print(f"Predicted: {predicted_min} | True: {true_next_min} | Error: {error:.4f}")
What IS genuinely proven
Keep these
Test Group 2 — depth ranking logic is sound
Synthetic valleys with known depths, radar measures curvature, ranking is compared. This is a legitimate test. The warning about width/curvature tradeoff shows good scientific thinking.
Level 4 — honesty tests are excellent
Testing convex problems where TerraLens should NOT help, testing hyperparameter sensitivity — this is real scientific integrity. Keep all of this.
EvolvingNet architecture — the right idea
The concept of adding neurons during training and re-initializing the optimizer is exactly correct. The execution just needs the map to actually guide the initialization.
96.6% MNIST accuracy — real result
TerraLens runs as a drop-in PyTorch optimizer and achieves real accuracy. This proves the system works end-to-end.
The honest version of your results
What to report
What you can claim now

TerraLens is a working optimizer. Depth ranking works. Satellite pruning works.

What you cannot claim yet

69.2% improvement. Map-guided init is proven superior.

What you need to fix

Group 1 must predict, not record. Group 3 must find the answer, not be given it.

After fixing — what you can claim

The real improvement number, which may be smaller but will be honest and publishable.

A real improvement of even 15-20% from a genuine test is far more valuable than a 69% result that a reviewer will immediately find the flaw in. Science requires the honest number.

Next steps — fix in this order

Fix first

Add predict_next_minimum() to EvolutionaryMap. Re-run Group 1.

Fix second

Remove hardcoded [2.0, 2.0]. Let TerraLens find it. Re-run Group 3.

Fix third

Compare TerraLens warm start vs Adam warm start (not just random cold start).

Then publish

Whatever number comes out of the fixed test — that is your real result.

Write fixed Group 3 ↗ Write predict_next_minimum ↗