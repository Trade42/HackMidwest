Step 4 runs after the test, on lane D, in the brief's order (`runs/after_test.sh`): the learned tracker
(`runs/learned.sh`), the tracker at s = 2e-3 and 5e-4 (`runs/extra.sh`), Z memory (`runs/extra.sh`), and the
tracker's cost per shot (`runs/timing.sh`). Each measures the tracker against the probes of its own validation
split (seeds 7240, 7260, 7220; training seeds 7140, 7160, 7120). None of them decodes.
