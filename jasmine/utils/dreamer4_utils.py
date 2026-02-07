import einops
import jax
import jax.numpy as jnp
from einops import rearrange


def patchify(videos: jax.Array, size: int) -> jax.Array:
    B, T, H, W, C = videos.shape
    x = jnp.pad(videos, ((0, 0), (0, 0), (0, -H % size), (0, -W % size), (0, 0)))
    return einops.rearrange(
        x, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size
    )

def unpatchify(patches: jax.Array, size: int, h_out: int, w_out: int) -> jax.Array:
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = einops.rearrange(
        patches,
        "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c",
        hp=size,
        wp=size,
        hn=hn,
    )
    return x[:, :, :h_out, :w_out]

def pack_bottleneck_to_spatial(z_btLd, *, n_spatial: int, k: int):
    """
    (B,T,N_b,D_b) -> (B,T,S_z, D_z_pre) by merging k tokens along N_b into channels.
    Requires: N_b == n_spatial * k  (e.g., 512 -> 256 with k=2).
    """
    return rearrange(z_btLd, 'b t (n_spatial k) d -> b t n_spatial (k d)', n_spatial=n_spatial, k=k)

def unpack_spatial_to_bottleneck(z_btLd, *, n_spatial: int, k: int):
    """
    (B,T,S_z, D_z_pre) -> (B,T,N_b,D_b) by splitting D_z_pre into k channels along N_b.
    Requires: N_b == n_spatial * k  (e.g., 256 -> 512 with k=2).
    """
    return rearrange(z_btLd, 'b t n_spatial (k d) -> b t (n_spatial k) d', n_spatial=n_spatial, k=k)