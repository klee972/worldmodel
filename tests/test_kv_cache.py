#!/usr/bin/env python3
"""Numerical correctness test: KV-cache path must match full-sequence forward pass."""
import sys
sys.path.insert(0, "/home/4bkang/rl/jasmine")

import jax
import jax.numpy as jnp
from flax import nnx

from jasmine.models.dreamer4_models import DynamicsDreamer4


# ── Shared model configuration ─────────────────────────────────────────────────
def _make_models(decode_infer: bool = True):
    """Return (train_model, infer_model) sharing the same initial weights.

    Both models are created from rngs seeded with the same value.  Because the
    decode flag only changes the forward path (not weight shapes or init order),
    the parameters are numerically identical.
    """
    kw = dict(
        d_model=64,
        d_spatial=32,
        n_spatial=4,
        n_register=2,
        n_agent=0,
        n_heads=4,
        n_actions=8,
        depth=8,
        k_max=8,
        time_every=4,
        pos_emb_type="rope",
    )
    train_model = DynamicsDreamer4(
        decode=False, rngs=nnx.Rngs(params=0, dropout=1), **kw
    )
    infer_model = DynamicsDreamer4(
        decode=decode_infer, rngs=nnx.Rngs(params=0, dropout=1), **kw
    )
    return train_model, infer_model


def _fill_context(model, actions_ctx, z_ctx, e, sig_clean):
    """Fill the model's KV cache frame-by-frame.

    Shifting is done here: frame t gets a_{t-1}, frame 0 gets sentinel -1.
    actions_ctx is the original (unshifted) context action sequence.
    """
    B, T_ctx = actions_ctx.shape
    step_idx_1 = jnp.full((B, 1), e, dtype=jnp.int32)
    sig_idx_1  = jnp.full((B, 1), sig_clean, dtype=jnp.int32)

    sentinel = jnp.full((B, 1), -1, dtype=actions_ctx.dtype)
    shifted  = jnp.concatenate([sentinel, actions_ctx[:, :-1]], axis=1)  # (B, T_ctx)
    for t in range(T_ctx):
        model(shifted[:, t:t+1], step_idx_1, sig_idx_1, z_ctx[:, t:t+1], deterministic=True)
        model.advance_cache()


# ── Test 1: frame-by-frame decode == full-sequence ─────────────────────────────
def test_kv_cache_correctness():
    B, T_ctx = 2, 4
    n_spatial, d_spatial, n_actions, k_max = 4, 32, 8, 8
    e = 0
    sig_clean = k_max - 1

    train_model, infer_model = _make_models()

    key = jax.random.PRNGKey(42)
    k1, k3, k4 = jax.random.split(key, 3)

    actions_ctx = jax.random.randint(k1, (B, T_ctx), 0, n_actions)
    z_ctx       = jax.random.normal(k3, (B, T_ctx, n_spatial, d_spatial))
    z_t         = jax.random.normal(k4, (B, 1,     n_spatial, d_spatial))

    step_idx_ctx = jnp.full((B, T_ctx), e,         dtype=jnp.int32)
    step_idx_fut = jnp.full((B, 1),     e,         dtype=jnp.int32)
    sig_ctx      = jnp.full((B, T_ctx), sig_clean, dtype=jnp.int32)
    sig_fut      = jnp.full((B, 1),     sig_clean, dtype=jnp.int32)

    # ── Full-sequence reference ──────────────────────────────────────────────
    # Shift externally: [-1, a0, ..., a_{T_ctx-1}] for T_ctx+1 frames
    sentinel      = jnp.full((B, 1), -1, dtype=actions_ctx.dtype)
    actions_full  = jnp.concatenate([sentinel, actions_ctx], axis=1)  # (B, T_ctx+1)
    step_idx_full = jnp.concatenate([step_idx_ctx, step_idx_fut], axis=1)
    sig_full      = jnp.concatenate([sig_ctx, sig_fut], axis=1)
    z_seq         = jnp.concatenate([z_ctx, z_t], axis=1)

    x1_hat_full, _ = train_model(actions_full, step_idx_full, sig_full, z_seq,
                                 deterministic=True)
    ref = x1_hat_full[:, -1:, :, :]  # (B, 1, n_spatial, d_spatial)

    # ── KV-cache path ────────────────────────────────────────────────────────
    infer_model.init_cache(B, T_ctx + 1)
    _fill_context(infer_model, actions_ctx, z_ctx, e, sig_clean)

    # Future frame's action token = a_{T_ctx-1} = actions_ctx[:, -1]
    x1_hat_kv, _ = infer_model(
        actions_ctx[:, -1:], step_idx_fut, sig_fut, z_t, deterministic=True
    )

    # ── Compare ──────────────────────────────────────────────────────────────
    max_diff = float(jnp.max(jnp.abs(x1_hat_kv - ref)))
    print(f"  max |kv_cache - full_seq|: {max_diff:.3e}")
    assert max_diff < 1e-4, f"FAILED: outputs diverge (max_diff={max_diff:.3e})"
    print("  PASSED")


# ── Test 2: incremental decode (rolling cache) == fresh fill from scratch ──────
def test_extend_kv_cache_correctness():
    """Running the decode model frame-by-frame automatically extends the cache.

    For each horizon frame t, the prediction from the incrementally-extended
    cache must match a prediction from a freshly initialised cache that is
    filled with all context + the first t future frames from scratch.
    """
    B, T_ctx, H = 2, 4, 3
    n_spatial, d_spatial, n_actions, k_max = 4, 32, 8, 8
    e = 0
    sig_clean = k_max - 1

    key = jax.random.PRNGKey(99)
    keys = jax.random.split(key, 10)

    actions_ctx = jax.random.randint(keys[0], (B, T_ctx), 0, n_actions)
    z_ctx       = jax.random.normal(keys[1],  (B, T_ctx, n_spatial, d_spatial))

    future_actions = [jax.random.randint(keys[2 + i],     (B, 1), 0, n_actions)
                      for i in range(H)]
    future_z       = [jax.random.normal(keys[2 + H + i],  (B, 1, n_spatial, d_spatial))
                      for i in range(H)]

    step_idx_1 = jnp.full((B, 1), e,         dtype=jnp.int32)
    sig_idx_1  = jnp.full((B, 1), sig_clean, dtype=jnp.int32)

    kw = dict(
        d_model=64,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=2,
        n_agent=0,
        n_heads=4,
        n_actions=n_actions,
        depth=8,
        k_max=k_max,
        time_every=4,
        pos_emb_type="rope",
        decode=True,
    )

    # ── Incremental path ─────────────────────────────────────────────────────
    infer_model = DynamicsDreamer4(rngs=nnx.Rngs(params=0, dropout=1), **kw)
    infer_model.init_cache(B, T_ctx + H)
    _fill_context(infer_model, actions_ctx, z_ctx, e, sig_clean)

    inc_preds = []
    prev_act_fut = actions_ctx[:, -1:]   # a_{T_ctx-1}: action token for frame T_ctx
    for t in range(H):
        pred, _ = infer_model(prev_act_fut, step_idx_1, sig_idx_1, future_z[t],
                              deterministic=True)
        infer_model.advance_cache()
        inc_preds.append(pred)
        prev_act_fut = future_actions[t]  # a_{T_ctx+t}: action token for next frame

    # ── Reference path (fresh init_cache before each frame) ──────────────────
    # Use a separate model object with the same initial weights (same seed).
    ref_model = DynamicsDreamer4(rngs=nnx.Rngs(params=0, dropout=1), **kw)

    ref_preds = []
    for t in range(H):
        # Reset the cache completely.
        ref_model.init_cache(B, T_ctx + H)

        # Fill T_ctx context frames.
        _fill_context(ref_model, actions_ctx, z_ctx, e, sig_clean)

        # Fill future frames 0 .. t-1.
        prev_act_ref = actions_ctx[:, -1:]
        for t2 in range(t):
            ref_model(prev_act_ref, step_idx_1, sig_idx_1, future_z[t2], deterministic=True)
            ref_model.advance_cache()
            prev_act_ref = future_actions[t2]

        # Decode frame t.
        pred_ref, _ = ref_model(prev_act_ref, step_idx_1, sig_idx_1, future_z[t],
                                deterministic=True)
        ref_preds.append(pred_ref)

    # ── Compare ──────────────────────────────────────────────────────────────
    for t in range(H):
        diff = float(jnp.max(jnp.abs(inc_preds[t] - ref_preds[t])))
        print(f"  frame {T_ctx + t}: max |incremental - ref|: {diff:.3e}")
        assert diff < 1e-4, f"FAILED at frame {T_ctx + t}: diff={diff:.3e}"
    print("  PASSED")


if __name__ == "__main__":
    print("=== KV cache correctness test ===")
    test_kv_cache_correctness()
    print()
    print("=== extend_kv_cache (rolling cache) correctness test ===")
    test_extend_kv_cache_correctness()
