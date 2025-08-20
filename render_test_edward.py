import os
import sys
from pathlib import Path
from typing import Optional

# Set environment variables for headless rendering
os.environ["MUJOCO_GL"] = "egl"
OS_ENV_PREALLOCATE = "XLA_PYTHON_CLIENT_PREALLOCATE"
if OS_ENV_PREALLOCATE not in os.environ:
  os.environ[OS_ENV_PREALLOCATE] = "false"

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
from tqdm import tqdm
from absl import app
from absl import flags

from mujoco_playground import registry
from mujoco_playground import wrapper

env_name = 'LeapDoorOpen'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Create an episode
rng = jax.random.PRNGKey(123)
rng, reset_rng = jax.random.split(rng)
state = jit_reset(reset_rng)

rollout = [state]
for step in tqdm(range(200)):
    action_size = env.mjx_model.nu
    action = jp.zeros(action_size)
    
    # Action[0] switches between -10 and 10 every 5 steps
    if (step // 5) % 2 == 0:
        action = action.at[0].set(10.0)
    else:
        action = action.at[0].set(-1.0)
    
    # Action[1] switches between -1 and 1 every 3 steps (different interval)
    if (step // 3) % 2 == 0:
        action = action.at[1].set(1.0)
    else:
        action = action.at[1].set(-1.0)
    
    # Action[2] oscillates between -0.5 and 0.5 every 4 steps
    if (step // 4) % 2 == 0:
        action = action.at[2].set(0.5)
    else:
        action = action.at[2].set(-0.5)
    
    # Action[3] oscillates between -0.5 and 0.5 every 7 steps
    if (step // 7) % 2 == 0:
        action = action.at[3].set(0.5)
    else:
        action = action.at[3].set(-0.5)
    
    # Action[4] oscillates between -0.5 and 0.5 every 11 steps
    if (step // 11) % 2 == 0:
        action = action.at[4].set(0.5)
    else:
        action = action.at[4].set(-0.5)
    
    state = jit_step(state, action)
    rollout.append(state)

render_every = 1
scene_option = mujoco.MjvOption()
frames = env.render(rollout[::render_every], height=480, width=640, camera="side")
media.write_video("test_env_collision.mp4", frames, fps=1.0 / env.dt / render_every)