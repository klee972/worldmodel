"""
CALVIN action bin distribution analyzer.

Loads rel_actions from CALVIN NPZ files and shows how many actions fall into
each bin under a given CALVINActionMapping configuration (uniform vs mu-law).

Usage:
    python analyze_calvin_action_bins.py --data_dir /path/to/calvin/training
    python analyze_calvin_action_bins.py --data_dir /path/to/calvin/training --use_mu_law --mu 5.0
    python analyze_calvin_action_bins.py --data_dir /path/to/calvin/training --n_arm_bins 21 --use_mu_law
"""

import argparse
import os
import numpy as np
from tqdm import tqdm



# ── inline copy of CALVINActionMapping so the script is self-contained ────────

class CALVINActionMapping:
    def __init__(self, n_arm_bins=11, arm_clip=1.0, use_mu_law=False, mu=5.0):
        self.n_arm_bins   = n_arm_bins
        self.arm_clip     = arm_clip
        self.use_mu_law   = use_mu_law
        self.mu           = mu
        self.n_gripper_bins = 2

    def _discretize_arm(self, x):
        x = np.clip(x, -self.arm_clip, self.arm_clip)
        if self.use_mu_law:
            x = x / self.arm_clip
            x = np.sign(x) * (np.log(1.0 + self.mu * np.abs(x)) / np.log(1.0 + self.mu))
            x = x * self.arm_clip
        normalized = (x + self.arm_clip) / (2.0 * self.arm_clip)
        bins = np.floor(normalized * self.n_arm_bins).astype(np.int32)
        return np.clip(bins, 0, self.n_arm_bins - 1)

    def _discretize_gripper(self, x):
        return (x >= 0).astype(np.int32)

    def continuous_to_indices(self, rel_actions):
        arm     = self._discretize_arm(rel_actions[..., :6])
        gripper = self._discretize_gripper(rel_actions[..., 6:7])
        return np.concatenate([arm, gripper], axis=-1).astype(np.int32)


# ── data loading ──────────────────────────────────────────────────────────────

def load_all_rel_actions(data_dir: str, max_frames: int | None = None) -> np.ndarray:
    """Load rel_actions from all episode NPZ files in data_dir."""
    ep_ids_path = os.path.join(data_dir, "ep_start_end_ids.npy")
    if not os.path.exists(ep_ids_path):
        raise FileNotFoundError(f"ep_start_end_ids.npy not found in {data_dir}")

    ep_ids = np.load(ep_ids_path)           # (N_episodes, 2)
    actions_list = []
    total = 0

    for start, end in tqdm(ep_ids):
        for frame_id in tqdm(range(int(start), int(end) + 1)):
            npz_path = os.path.join(data_dir, f"episode_{frame_id:07d}.npz")
            try:
                with np.load(npz_path) as f:
                    actions_list.append(f["rel_actions"].copy())   # (7,)
            except Exception as e:
                print(f"  Warning: skipping {npz_path}: {e}")
                continue
            total += 1
            if max_frames is not None and total >= max_frames:
                break
        if max_frames is not None and total >= max_frames:
            break

    print(f"Loaded {total:,} frames from {data_dir}")
    return np.stack(actions_list, axis=0)   # (N, 7)


# ── visualization ─────────────────────────────────────────────────────────────

DIM_NAMES = ["pos_x", "pos_y", "pos_z", "ori_a", "ori_b", "ori_c", "gripper"]

def print_bin_distribution(indices: np.ndarray, mapping: CALVINActionMapping) -> None:
    """Print per-dimension bin counts and a small ASCII bar chart."""
    N = indices.shape[0]
    bar_width = 40

    for dim in range(7):
        col = indices[:, dim]
        n_bins = 2 if dim == 6 else mapping.n_arm_bins
        counts = np.bincount(col, minlength=n_bins)

        print(f"\n── dim {dim}: {DIM_NAMES[dim]}  (n_bins={n_bins}, N={N:,}) ──")

        if dim == 6:
            # gripper: just two rows
            labels = ["close (0)", "open  (1)"]
            for b in range(2):
                pct = counts[b] / N * 100
                bar = "█" * int(pct / 100 * bar_width)
                print(f"  {labels[b]}  {counts[b]:>8,}  ({pct:5.1f}%)  {bar}")
        else:
            # arm dims: compact table with bin centre value
            print(f"  {'bin':>4}  {'centre':>8}  {'count':>8}  {'%':>6}  histogram")
            for b in range(n_bins):
                # recover bin centre in original space (before mu-law)
                # bin b covers [b/n, (b+1)/n) of the compressed domain
                centre_norm = (b + 0.5) / n_bins          # [0, 1]
                centre_comp = centre_norm * 2 - 1          # [-1, 1] in compressed space
                if mapping.use_mu_law:
                    # invert mu-law: x = sign(y)*(1/mu)*((1+mu)^|y| - 1)
                    centre_orig = (
                        np.sign(centre_comp)
                        * (1.0 / mapping.mu)
                        * ((1.0 + mapping.mu) ** abs(centre_comp) - 1.0)
                    )
                else:
                    centre_orig = centre_comp               # no compression
                centre_orig *= mapping.arm_clip             # scale back

                pct = counts[b] / N * 100
                bar = "█" * int(pct / 100 * bar_width)
                print(f"  {b:>4}  {centre_orig:>+8.4f}  {counts[b]:>8,}  {pct:>5.1f}%  {bar}")

    # summary: per-dim entropy (bits)
    print("\n── per-dimension entropy ──")
    for dim in range(7):
        col = indices[:, dim]
        n_bins = 2 if dim == 6 else mapping.n_arm_bins
        counts = np.bincount(col, minlength=n_bins).astype(float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_bins)
        print(f"  {DIM_NAMES[dim]:8s}  entropy={entropy:.3f} bits  "
              f"(max={max_entropy:.3f}, utilization={entropy/max_entropy*100:.1f}%)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CALVIN action bin distribution analyzer")
    parser.add_argument("--data_dir", type=str,
                        default="/home/4bkang/rl/calvin/dataset/task_ABCD_D/validation",
                        help="Path to CALVIN data directory (contains ep_start_end_ids.npy)")
    parser.add_argument("--n_arm_bins", type=int, default=21,
                        help="Number of uniform bins for each arm dimension (default: 11)")
    parser.add_argument("--arm_clip", type=float, default=1.0,
                        help="Clip rel_actions to [-arm_clip, arm_clip] (default: 1.0)")
    parser.add_argument("--use_mu_law", action="store_true",
                        help="Enable mu-law (foveated) quantization")
    parser.add_argument("--mu", type=float, default=10.0,
                        help="Mu-law compression strength (default: 5.0)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit number of frames to load (default: all)")
    args = parser.parse_args()

    mapping = CALVINActionMapping(
        n_arm_bins=args.n_arm_bins,
        arm_clip=args.arm_clip,
        use_mu_law=args.use_mu_law,
        mu=args.mu,
    )

    print("=" * 60)
    print("CALVINActionMapping configuration:")
    print(f"  n_arm_bins = {mapping.n_arm_bins}")
    print(f"  arm_clip   = {mapping.arm_clip}")
    print(f"  use_mu_law = {mapping.use_mu_law}")
    print(f"  mu         = {mapping.mu}")
    print("=" * 60)

    rel_actions = load_all_rel_actions(args.data_dir, max_frames=args.max_frames)

    print(f"\nrel_actions stats (before discretization):")
    for dim in range(7):
        col = rel_actions[:, dim]
        print(f"  {DIM_NAMES[dim]:8s}  "
              f"min={col.min():+.4f}  max={col.max():+.4f}  "
              f"mean={col.mean():+.4f}  std={col.std():.4f}")

    indices = mapping.continuous_to_indices(rel_actions)   # (N, 7)

    print_bin_distribution(indices, mapping)


if __name__ == "__main__":
    main()
