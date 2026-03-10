"""Quick demo: generate synthetic data, run Part 2 optimisation, and play back."""

from config import N, K
from data import generate_dataset
from part1 import baseline
from part2 import optimize
from viewer import play

traj_in, hotspots = generate_dataset(n=N, k=K)
init_pos          = baseline(traj_in)
pos_opt, _        = optimize(traj_in, init_pos=init_pos, hotspots=hotspots, n_steps=200)

play(pos_opt, title="Demo — Part 2 Optimised", hotspots=hotspots)
