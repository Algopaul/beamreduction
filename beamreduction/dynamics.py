from pathlib import Path

import jax
import jax.numpy as jnp


def spatial_grid(N: int = 101, beam_length: float = 1.0):
  x = jnp.linspace(0.0, beam_length, N)
  dx = float(x[1] - x[0])
  return x, dx


def timesteps(final_time, dt):
  return jnp.arange(0, final_time, dt)


# Time stepping (explicit for w_tt = -w_xxxx)
# dt = 0.2 * dx**2  # VERY important: dt ~ O(dx^2)
# T = 4.0
# steps = int(T / dt)
# print(steps // 400)
#
# outdir = Path("beam_frames")
# outdir.mkdir(exist_ok=True)


def fourth_derivative(u, dx):
  """
    4th derivative with 5-point stencil on interior nodes.
    Boundary entries left as zero; BCs are enforced separately.
    """
  out = jnp.zeros_like(u)
  stencil = (u[:-4] - 4.0 * u[1:-3] + 6.0 * u[2:-2] - 4.0 * u[3:-1] + u[4:])
  out = out.at[2:-2].set(stencil / dx**4)
  return out


def apply_bc(u):
  """
    Clamped at left: w(0)=0, w'(0)=0  -> w[0]=0, w[1]=0
    Free at right: w''(L)=0, w'''(L)=0
      -> w[-2] = 2*w[-3] - w[-4]
         w[-1] = 3*w[-3] - 2*w[-4]
    """
  u = u.at[0].set(0.0)
  u = u.at[1].set(0.0)

  u_nm3 = u[-3]
  u_nm4 = u[-4]
  u_nm2 = 2.0 * u_nm3 - u_nm4
  u_nm1 = 3.0 * u_nm3 - 2.0 * u_nm4

  u = u.at[-2].set(u_nm2)
  u = u.at[-1].set(u_nm1)
  return u


def get_step(c2, dx, dt):

  @jax.jit
  def step(i, state):
    del i
    w, w_prev = state
    acc = -c2 * fourth_derivative(w, dx)
    w_next = 2.0 * w - w_prev + dt**2 * acc
    w_next = apply_bc(w_next)
    return w_next, w

  return step


# Initial deflection:
# smooth shape satisfying left clamp; free-end BCs will be enforced via apply_bc


def integrate(w0, step_fn, n_steps):

  w, w_prev = w0, w0

  def body_fn(state, t):
    del t
    w, w_prev = state
    out = jax.lax.fori_loop(0, 500, step_fn, (w, w_prev))
    return out, out

  _, scout = jax.lax.scan(body_fn, (w, w_prev), jnp.arange(n_steps))
  return scout[0]


# out = integrate(w0)
#
# for i, y in enumerate(out):
#   plt.clf()
#   plt.plot(x, y, lw=2) @ VT[0, i]
#   ut
#   plt.ylim(-0.12, 0.12)
#   plt.xlabel("x")
#   plt.ylabel("deflection")
#   # plt.title(f"t = {n * dt:.4f}")
#   plt.grid(alpha=0.3)
#   plt.savefig(outdir / f"frame_{i:05d}.png", dpi=150)
#
# U, S, VT = jnp.linalg.svd(out.T)
# for i in range(10):
#   plt.clf()
#   plt.plot(x, U[:, i], lw=2)
#   plt.ylim(-0.32, 0.32)
#   plt.xlabel("x")
#   plt.ylabel("mode")
#   # plt.title(f"t = {n * dt:.4f}")
#   plt.grid(alpha=0.3)
#   plt.savefig(outdir / f"mode_{i:05d}.png", dpi=150)
#
#
# def modal_reconstruction(i, rmax=3):
#   U[:, 0] @ VT[0, i]
#   return U[:, :rmax] @ jnp.diag(S[:rmax]) @ VT[:rmax, i]
#
#
# for i, y in enumerate(out):
#   plt.clf()
#   plt.plot(x, y, lw=2)
#   plt.plot(x, modal_reconstruction(i, 1), lw=2)
#   # plt.plot(x, modal_reconstruction(i, 2), lw=2)
#   # plt.plot(x, modal_reconstruction(i, 3), lw=2)
#   plt.legend(['true', 'r1', 'r2', 'r3'])
#   plt.ylim(-0.12, 0.12)
#   plt.xlabel("x")
#   plt.ylabel("deflection")
#   # plt.title(f"t = {n * dt:.4f}")
#   plt.grid(alpha=0.3)
#   plt.savefig(outdir / f"modal_1_{i:05d}.png", dpi=150)
#
# for i, y in enumerate(out):
#   plt.clf()
#   plt.plot(x, y, lw=2)
#   plt.plot(x, modal_reconstruction(i, 1), lw=2)
#   plt.plot(x, modal_reconstruction(i, 2), lw=2)
#   # plt.plot(x, modal_reconstruction(i, 3), lw=2)
#   plt.legend(['true', 'r1', 'r2', 'r3'])
#   plt.ylim(-0.12, 0.12)
#   plt.xlabel("x")
#   plt.ylabel("deflection")
#   # plt.title(f"t = {n * dt:.4f}")
#   plt.grid(alpha=0.3)
#   plt.savefig(outdir / f"modal_2_{i:05d}.png", dpi=150)
