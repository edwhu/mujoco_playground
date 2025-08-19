#!/usr/bin/env python3
"""Render a simple collision test for LeapDoorOpen environment."""

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
from absl import app
from absl import flags

from mujoco_playground import registry
from mujoco_playground import wrapper

# Define flags
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "collision_test.mp4", "Output video path"
)
_EPISODE_LENGTH = flags.DEFINE_integer(
    "episode_length", 30, "Episode length for rendering"
)
_RENDER_EVERY = flags.DEFINE_integer(
    "render_every", 1, "Render every N steps"
)
_ACTION_MAGNITUDE = flags.DEFINE_float(
    "action_magnitude", 0.9, "Magnitude of forward movement action (will be clipped to control ranges)"
)


def create_forward_action_sequence(env, num_actions: int, magnitude: float):
    """Create a sequence of actions that move the MCP joints to their maximum values."""
    # Get the action space size from the environment's MJX model
    action_size = env.mjx_model.nu
    
    # Create a fixed action that moves MCP joints to their maximum values
    # The action should be in the range [-1, 1] and will be scaled by action_scale
    # With action_scale=0.6, a magnitude of 1.0 results in a motor target offset of 0.6
    # Recommended magnitude: 0.1-0.5 for gentle movement, 0.5-1.0 for faster movement
    forward_action = jp.zeros(action_size)
    
    # Set MCP joints to maximum values (positive magnitude)
    # These are the MCP (metacarpophalangeal) joints for index, middle, and ring fingers
    forward_action = forward_action.at[5].set(magnitude)   # if_mcp_act (index finger MCP)
    forward_action = forward_action.at[9].set(magnitude)   # mf_mcp_act (middle finger MCP) 
    forward_action = forward_action.at[13].set(magnitude)  # rf_mcp_act (ring finger MCP)
    
    # Repeat this action for the sequence
    actions = jp.tile(forward_action, (num_actions, 1))
    
    return actions


def render_collision_test(env, action_sequence, episode_length: int, render_every: int = 1, camera: str = "side"):
    """Render a collision test using fixed actions.

    Args:
      env: environment to test.
      action_sequence: sequence of actions to apply.
      episode_length: max steps per episode.
      render_every: subsample cadence for rendering.
      camera: name of the camera to use for rendering.
    Returns:
      frames: list of RGB frames.
      fps: frames per second used for the video.
    """
    
    # JIT compile the functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # RNG
    rng = jax.random.PRNGKey(123)
    
    # For evaluation we use a single environment
    num_envs = 1
    print(f"Using {num_envs} environments for collision test")
    
    rollout = []
    print(f"Starting collision test, episode length: {episode_length}")
    
    # Reset environment
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    state0 = state
    rollout.append(state0)

    for step in range(episode_length):
        # Use the fixed action from our sequence
        action = action_sequence[step]
        
        # Debug: Print action and motor targets for first few steps
        state = jit_step(state, action)
        state0 = state
        rollout.append(state0)
        
        # Logging
        if hasattr(state0.reward, 'shape') and len(state0.reward.shape) > 0:
            reward_mean = jp.mean(state0.reward).item()
            done_mean = jp.mean(state0.done).item()
            print(f"Step {step + 1}: reward={reward_mean:.4f}, done={done_mean:.4f}")
        else:
            print(f"Step {step + 1}: reward={state0.reward:.4f}, done={state0.done}")
        
        # Stop early if episode terminates
        if hasattr(state0.done, 'shape') and len(state0.done.shape) > 0:
            if jp.any(state0.done):
                print(f"Episode terminated at step {step + 1}")
                break
        else:
            if state0.done:
                print(f"Episode terminated at step {step + 1}")
                break
    
    print(f"Collision test completed with {len(rollout)} states")
    
    # Render the rollout
    print("Rendering video...")
    fps = 1.0 / env.dt / render_every
    print(f"FPS: {fps}")
    
    # Sample frames for rendering
    traj = rollout[::render_every]
    print(f"Rendering {len(traj)} frames")
    
    # Set up scene options
    scene_option = mujoco.MjvOption()
    # Ensure all geom groups are visible (0-5)
    try:
        scene_option.geomgroup[:] = 1
    except Exception:
        # Fallback for older mujoco versions
        for i in range(len(scene_option.geomgroup)):
            scene_option.geomgroup[i] = 1
    # Disable transparency so all objects render solid
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    
    # Render frames with specified camera
    frames = env.render(
        traj, height=480, width=640, scene_option=scene_option, camera=camera
    )
    
    return frames, fps


def main(argv):
    """Main function."""
    del argv  # Unused
    
    env_name = "LeapDoorOpen"
    output_path = _OUTPUT_PATH.value
    episode_length = _EPISODE_LENGTH.value
    render_every = _RENDER_EVERY.value
    action_magnitude = _ACTION_MAGNITUDE.value
    
    print(f"Environment: {env_name}")
    print(f"Output path: {output_path}")
    print(f"Episode length: {episode_length}")
    print(f"Render every: {render_every} steps")
    print(f"Action magnitude: {action_magnitude}")
    
    try:
        # Load environment
        env_cfg = registry.get_default_config(env_name)
        env = registry.load(env_name, config=env_cfg)
        print(f"Environment loaded successfully")
        
        # Create action sequence
        action_sequence = create_forward_action_sequence(env, episode_length, action_magnitude)
        print(f"Created action sequence with {len(action_sequence)} actions")
        
        # Render collision test with "fixed" camera
        frames, fps = render_collision_test(env, action_sequence, episode_length, render_every, camera="fixed")
        
        # Save video
        media.write_video(output_path, frames, fps=fps)
        print(f"Video saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    app.run(main) 