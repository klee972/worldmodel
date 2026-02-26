#!/usr/bin/env python3
"""Numerical correctness test: KV-cache path must match full-sequence forward pass."""
import sys
sys.path.insert(0, "/home/4bkang/rl/jasmine")

import jax
import jax.numpy as jnp
from flax import nnx

from jasmine.models.dreamer4_models import DynamicsDreamer4


def test_kv_cache_correctness():
    # Small model — depth=8 with time_every=4 → 2 temporal attention layers
    B, T_ctx = 2, 4
    d_model   = 64
    n_spatial = 4
    d_spatial = 32
    n_actions = 8
    depth     = 8
    time_every = 4
    n_heads   = 4
    k_max     = 8

    rngs = nnx.Rngs(params=0, dropout=1)
    model = DynamicsDreamer4(
        d_model=d_model,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=2,
        n_agent=0,
        n_heads=n_heads,
        n_actions=n_actions,
        depth=depth,
        k_max=k_max,
        rngs=rngs,
        time_every=time_every,
        pos_emb_type="rope",
        shift_action_tokens_by_one=True,
    )

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    actions_ctx  = jax.random.randint(k1, (B, T_ctx), 0, n_actions)
    action_curr  = jax.random.randint(k2, (B, 1),     0, n_actions)
    z_ctx        = jax.random.normal(k3, (B, T_ctx, n_spatial, d_spatial))
    z_t          = jax.random.normal(k4, (B, 1,     n_spatial, d_spatial))

    e = 0  # step_idx for d=1/k_max (finest)
    step_idx_ctx = jnp.full((B, T_ctx), e,         dtype=jnp.int32)
    step_idx_fut = jnp.full((B, 1),     e,         dtype=jnp.int32)
    sig_ctx      = jnp.full((B, T_ctx), k_max - 1, dtype=jnp.int32)
    sig_fut      = jnp.full((B, 1),     k_max - 1, dtype=jnp.int32)

    # ── Full-sequence path ──────────────────────────────────────────────
    actions_full  = jnp.concatenate([actions_ctx, action_curr], axis=1)   # (B, T_ctx+1)
    step_idx_full = jnp.concatenate([step_idx_ctx, step_idx_fut], axis=1)
    sig_full      = jnp.concatenate([sig_ctx, sig_fut], axis=1)
    z_seq         = jnp.concatenate([z_ctx, z_t], axis=1)                 # (B, T_ctx+1, ...)

    x1_hat_full, _ = model(actions_full, step_idx_full, sig_full, z_seq)
    ref = x1_hat_full[:, -1:, :, :]   # (B, 1, n_spatial, d_spatial)

    # ── KV-cache path ───────────────────────────────────────────────────
    kv_cache = model.encode_context(actions_ctx, step_idx_ctx, sig_ctx, z_ctx)

    # Pass [a_{T_ctx-1}, a_curr] so after shift-by-one, pos 1 = emb(a_{T_ctx-1})
    actions_for_decode = jnp.concatenate([actions_ctx[:, -1:], action_curr], axis=1)
    x1_hat_kv, _ = model(
        actions_for_decode, step_idx_fut, sig_fut, z_t,
        kv_cache=kv_cache,
    )

    # ── Compare ─────────────────────────────────────────────────────────
    max_diff = float(jnp.max(jnp.abs(x1_hat_kv - ref)))
    print(f"  max |kv_cache - full_seq|: {max_diff:.3e}")
    assert max_diff < 1e-4, f"FAILED: outputs diverge (max_diff={max_diff:.3e})"
    print("  PASSED")


def test_extend_kv_cache_correctness():
    """extend_kv_cache(ctx) + decode(frame t+1) must match full-seq encode_context(ctx+t+1) + decode."""
    B, T_ctx, H = 2, 4, 3   # H = number of horizon frames to test
    d_model   = 64
    n_spatial = 4
    d_spatial = 32
    n_actions = 8
    depth     = 8
    time_every = 4
    n_heads   = 4
    k_max     = 8

    rngs = nnx.Rngs(params=0, dropout=1)
    model = DynamicsDreamer4(
        d_model=d_model,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=2,
        n_agent=0,
        n_heads=n_heads,
        n_actions=n_actions,
        depth=depth,
        k_max=k_max,
        rngs=rngs,
        time_every=time_every,
        pos_emb_type="rope",
        shift_action_tokens_by_one=True,
    )

    key = jax.random.PRNGKey(99)
    keys = jax.random.split(key, 10)

    e = 0
    sig_clean = k_max - 1

    actions_ctx = jax.random.randint(keys[0], (B, T_ctx), 0, n_actions)
    z_ctx       = jax.random.normal(keys[1], (B, T_ctx, n_spatial, d_spatial))

    # Generate H future frames and actions
    future_actions = [jax.random.randint(keys[2+i], (B, 1), 0, n_actions) for i in range(H)]
    future_z       = [jax.random.normal(keys[2+H+i], (B, 1, n_spatial, d_spatial)) for i in range(H)]

    # ── Incremental (extend_kv_cache) path ─────────────────────────────
    kv_cache = model.encode_context(
        actions_ctx,
        jnp.full((B, T_ctx), e, dtype=jnp.int32),
        jnp.full((B, T_ctx), sig_clean, dtype=jnp.int32),
        z_ctx,
    )

    inc_ctx_actions = actions_ctx
    inc_preds = []
    for t in range(H):
        action_curr = future_actions[t]
        z_t         = future_z[t]

        # Decode using current growing cache
        actions_for_decode = jnp.concatenate([inc_ctx_actions[:, -1:], action_curr], axis=1)
        x1_hat_inc, _ = model(
            actions_for_decode,
            jnp.full((B, 1), e, dtype=jnp.int32),
            jnp.full((B, 1), sig_clean, dtype=jnp.int32),
            z_t,
            kv_cache=kv_cache,
        )
        inc_preds.append(x1_hat_inc)

        # Extend cache with z_t (simulating the denoised frame being appended)
        actions_for_extend = jnp.concatenate([inc_ctx_actions[:, -1:], action_curr], axis=1)
        kv_cache = model.extend_kv_cache(
            actions_for_extend,
            jnp.full((B, 1), e, dtype=jnp.int32),
            jnp.full((B, 1), sig_clean, dtype=jnp.int32),
            z_t,
            kv_cache,
        )
        inc_ctx_actions = jnp.concatenate([inc_ctx_actions, action_curr], axis=1)

    # ── Reference (encode_context from scratch each time) path ─────────
    ref_ctx_actions = actions_ctx
    ref_ctx_z       = z_ctx
    ref_preds = []
    for t in range(H):
        action_curr = future_actions[t]
        z_t         = future_z[t]

        T_ref = ref_ctx_z.shape[1]
        kv_ref = model.encode_context(
            ref_ctx_actions,
            jnp.full((B, T_ref), e, dtype=jnp.int32),
            jnp.full((B, T_ref), sig_clean, dtype=jnp.int32),
            ref_ctx_z,
        )
        actions_for_decode = jnp.concatenate([ref_ctx_actions[:, -1:], action_curr], axis=1)
        x1_hat_ref, _ = model(
            actions_for_decode,
            jnp.full((B, 1), e, dtype=jnp.int32),
            jnp.full((B, 1), sig_clean, dtype=jnp.int32),
            z_t,
            kv_cache=kv_ref,
        )
        ref_preds.append(x1_hat_ref)
        ref_ctx_z       = jnp.concatenate([ref_ctx_z, z_t], axis=1)
        ref_ctx_actions = jnp.concatenate([ref_ctx_actions, action_curr], axis=1)

    # ── Compare ─────────────────────────────────────────────────────────
    for t in range(H):
        diff = float(jnp.max(jnp.abs(inc_preds[t] - ref_preds[t])))
        print(f"  frame {T_ctx+t}: max |extend_kv - ref|: {diff:.3e}")
        assert diff < 1e-4, f"FAILED at frame {T_ctx+t}: diff={diff:.3e}"
    print("  PASSED")


if __name__ == "__main__":
    print("=== KV cache correctness test ===")
    test_kv_cache_correctness()
    print()
    print("=== extend_kv_cache correctness test ===")
    test_extend_kv_cache_correctness()
