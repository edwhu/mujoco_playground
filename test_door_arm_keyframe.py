#!/usr/bin/env python3
"""Test script to render the door arm environment and check keyframe initialization."""

import os
import sys
from pathlib import Path

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

def test_door_arm_keyframe():
    """Test the door arm keyframe initialization."""
    
    env_name = 'LeapDoorOpenArm'
    print(f"Loading environment: {env_name}")
    
    try:
        env = registry.load(env_name)
        env_cfg = registry.get_default_config(env_name)
        print(f"Environment loaded successfully")
        
        # Print environment info
        print(f"Action size: {env.action_size}")
        print(f"Number of actuators (nu): {env.mjx_model.nu}")
        print(f"Number of joints (nv): {env.mjx_model.nv}")
        print(f"Number of position states (nq): {env.mjx_model.nq}")
        
        # Create an episode
        rng = jax.random.PRNGKey(123)
        rng, reset_rng = jax.random.split(rng)
        
        print("Resetting environment...")
        state = env.reset(reset_rng)
        print(f"Environment reset successfully")
        
        # Print initial state info
        print(f"Initial state obs shape: {state.obs['state'].shape}")
        print(f"Initial state privileged shape: {state.obs['privileged_state'].shape}")
        
        # Render the first frame
        print("Rendering first frame...")
        frames = env.render([state], height=480, width=640, camera="side")
        
        # Save the image
        output_path = "door_arm_initial_frame.png"
        media.write_image(output_path, frames[0])
        print(f"Saved initial frame to: {output_path}")
        
        # Also save as video with just one frame for debugging
        video_path = "door_arm_initial_frame.mp4"
        media.write_video(video_path, frames, fps=1.0)
        print(f"Saved initial frame video to: {video_path}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_door_arm_keyframe() 