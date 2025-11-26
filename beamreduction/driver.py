import logging

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from omegaconf import OmegaConf

from beamreduction import dynamics as dn
from beamreduction.config import Config
from beamreduction.util import log_duration


def store_animation(x, data, filename, labels):
  # Main animation (only full order model)
  fig, ax = plt.subplots()
  lines = []
  for d in data:
    (line,) = ax.plot(x, d[0])
    lines.append(line)

  ax.set_ylim((-0.1, 0.1))
  ax.legend(labels, loc='upper left')
  ax.set_ylabel('deformation')

  def fupdate(frame):
    for l, d in zip(lines, data):
      l.set_ydata(d[frame])
    return line,

  ani = FuncAnimation(fig, fupdate, frames=len(data[0]), interval=10, blit=True)
  try:
    ani.save(
        filename + ".mp4",
        writer="ffmpeg",
        fps=30,
        dpi=200,
    )
    pass
  except Exception as e:
    logging.warning('No ffmpeg available. For mp4 output install ffmpeg')
    try:
      ani.save(
          filename + ".gif",
          writer="pillow",
          fps=15,
      )
      pass
    except Exception as e:
      logging.warning('No pillow available. For gif output install pillo')


@hydra.main(version_base=None, config_name='config', config_path='../conf')
@log_duration()
def main(cfg: Config) -> None:
  logging.info("\n%s", OmegaConf.to_yaml(cfg))
  x, dx = dn.spatial_grid()
  dt = 0.1 * dx**2
  n_time_steps = 400
  c2 = 3.0  # scaled EI/(rho A)

  # Initial deformation
  w0 = 0.1 * (1.0 - jnp.cos(jnp.pi * x / 2.0))**2
  w0 = dn.apply_bc(w0)
  step_fn = dn.get_step(c2, dx, dt)
  logging.info('Start simulation')
  trajectory = dn.integrate(w0, step_fn, n_time_steps)
  jax.block_until_ready(trajectory)
  logging.info('Simulation complete')

  logging.info('Computing SVD of simulation data')
  U, S, VT = jnp.linalg.svd(trajectory.T, full_matrices=False)

  fig, ax = plt.subplots()
  ax.semilogy(S / S[0])
  ax.legend(['normalized singular values'])
  plt.savefig('singular_values.png')
  plt.close(fig)

  for mode in range(5):
    fig, ax = plt.subplots()
    ax.plot(U[:, mode])
    ax.legend([f'mode {mode+1}'])
    plt.savefig(f'mode_{mode+1:02d}.png')
    plt.close(fig)

  store_animation(x, [trajectory], 'full_model', ['full trajectory'])

  reduced_trajectories = []
  labels = ['reduced dim=' + str(x) for x in range(1, 4)]
  for r in range(1, 4):
    reduced_trajectories.append((U[:, :r] @ jnp.diag(S[:r]) @ VT[:r, :]).T)

  for r in range(1, 4):
    store_animation(
        x,
        [trajectory, *reduced_trajectories[:r]],
        f'reduced_{r}',
        ['full trajectory', *labels[:r]],
    )

  pass


if __name__ == "__main__":
  main()
