"""Quick demo: generate synthetic data, run Part 1, then Part 2, and play back."""

from config import N, K
from data import generate_dataset
from part1 import bundle_trajectories, assign_offsets
from part2 import optimize
from viewer import play

traj_in, hotspots = generate_dataset(n=N, k=K)
bundled, groups   = bundle_trajectories(traj_in, hotspots=hotspots)
init_pos          = assign_offsets(bundled, groups)
pos_opt, _        = optimize(traj_in, init_pos=init_pos, hotspots=hotspots, n_steps=200)

play(pos_opt, title="Demo — Part 2 Optimised")
