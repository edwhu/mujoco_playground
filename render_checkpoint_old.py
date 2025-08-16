#!/usr/bin/env python3
"""Render a trained model checkpoint as a video."""

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
from orbax import checkpoint as ocp
from absl import app
from absl import flags

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import dm_control_suite_params
from ml_collections import config_dict

# Define flags
_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir", None, "Path to checkpoint directory", required=True
)
_ENV_NAME = flags.DEFINE_string(
    "env_name", "LeapCubeRotateZAxis", "Environment name"
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "rollout.mp4", "Output video path"
)
_EPISODE_LENGTH = flags.DEFINE_integer(
    "episode_length", 500, "Episode length for rendering"
)
_RENDER_EVERY = flags.DEFINE_integer(
    "render_every", 1, "Render every N steps"
)
_NUM_EPISODES = flags.DEFINE_integer(
    "num_episodes", 1, "Number of episodes to render into one video"
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    """Get the RL configuration for the environment."""
    import mujoco_playground
    
    if env_name in mujoco_playground.manipulation._envs:
        return manipulation_params.brax_ppo_config(env_name)
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_ppo_config(env_name)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        return dm_control_suite_params.brax_ppo_config(env_name)

    raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def get_latest_checkpoint_path(checkpoint_dir: str):
    """Get the path to the latest checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Check if there's a checkpoints subdirectory
    checkpoints_path = checkpoint_path / "checkpoints"
    if checkpoints_path.exists():
        checkpoint_path = checkpoints_path
        print(f"Found checkpoints subdirectory: {checkpoints_path}")
    
    # Find the latest checkpoint
    checkpoint_dirs = [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.isdigit()]
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {checkpoint_path}")
    
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name))
    print(f"Loading checkpoint from: {latest_checkpoint}")
    
    return str(latest_checkpoint)


def load_checkpoint_and_create_inference_fn(checkpoint_path, env_name):
    """Load checkpoint and create inference function using the same approach as training script."""
    from brax.training.agents.ppo import train as ppo
    
    # Get the RL config for the environment
    ppo_params = get_rl_config(env_name)
    
    # Create training parameters (same as in train_jax_ppo.py)
    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]
    if "num_timesteps" in training_params:
        del training_params["num_timesteps"]
    
    # Create network factory
    from brax.training.agents.ppo import networks as ppo_networks
    network_fn = ppo_networks.make_ppo_networks
    if hasattr(ppo_params, "network_factory"):
        import functools
        network_factory = functools.partial(
            network_fn, **ppo_params.network_factory
        )
    else:
        network_factory = network_fn
    
    # Create a dummy environment for the training function
    # We only need the make_inference_fn, not the actual training
    def dummy_progress_fn(num_steps, metrics):
        pass
    
    def dummy_policy_params_fn(current_step, make_policy, params):
        pass
    
    # Create a dummy environment (we'll use the real one for inference)
    from mujoco_playground import registry
    env_cfg = registry.get_default_config(env_name)
    dummy_env = registry.load(env_name, config=env_cfg)
    
    # Get the make_inference_fn from the training function with checkpoint restoration
    make_inference_fn, params, _ = ppo.train(
        environment=dummy_env,
        progress_fn=dummy_progress_fn,
        policy_params_fn=dummy_policy_params_fn,
        eval_env=dummy_env,
        restore_checkpoint_path=checkpoint_path,
        num_timesteps=0,  # Don't train, just load
        **training_params,  # Include all PPO parameters
        network_factory=network_factory,
        seed=0,  # Use a fixed seed
        wrap_env_fn=wrapper.wrap_for_brax_training,  # Wrap environment properly
    )
    
    # Create the inference function
    inference_fn = make_inference_fn(params, deterministic=True)
    
    return inference_fn, params


def render_rollout(env, inference_fn, episode_length: int, render_every: int = 2, num_episodes: int = 1):
    """Render one or more episodes using the trained model, concatenated.

    Args:
      env: wrapped evaluation environment (same wrapping as training).
      inference_fn: compiled policy function.
      episode_length: max steps per episode before forcing reset.
      render_every: subsample cadence for rendering.
      num_episodes: number of episodes to record back-to-back.
    Returns:
      frames: list of RGB frames.
      fps: frames per second used for the video.
    """
    
    # JIT compile the functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)
    
    # RNG
    rng = jax.random.PRNGKey(123)
    
    # For evaluation we use a single environment
    num_envs = 1
    print(f"Using {num_envs} environments for evaluation")
    
    rollout = []
    print(f"Starting rendering for {num_episodes} episode(s), episode length: {episode_length}")
    
    for ep in range(num_episodes):
        # Reset each episode to ensure fresh randomization
        rng, reset_rng = jax.random.split(rng)
        reset_rng = jp.asarray(jax.random.split(reset_rng, num_envs))
        state = jit_reset(reset_rng)
        state0 = state
        rollout.append(state0)

        for step in range(episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            state0 = state
            rollout.append(state0)
            
            # Logging
            if hasattr(state0.reward, 'shape') and len(state0.reward.shape) > 0:
                reward_mean = jp.mean(state0.reward).item()
                done_mean = jp.mean(state0.done).item()
                print(f"Ep {ep+1} Step {step + 1}: reward={reward_mean:.4f}, done={done_mean:.4f}")
            else:
                print(f"Ep {ep+1} Step {step + 1}: reward={state0.reward:.4f}, done={state0.done}")
            
            # Stop early if episode terminates
            if hasattr(state0.done, 'shape') and len(state0.done.shape) > 0:
                if jp.any(state0.done):
                    print(f"Episode {ep+1} terminated at step {step + 1}")
                    break
            else:
                if state0.done:
                    print(f"Episode {ep+1} terminated at step {step + 1}")
                    break
    
    print(f"Rollout completed with {len(rollout)} states across {num_episodes} episode(s)")
    
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
    
    # Render frames
    frames = env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    
    return frames, fps


def main(argv):
    """Main function."""
    del argv  # Unused
    
    checkpoint_dir = _CHECKPOINT_DIR.value
    env_name = _ENV_NAME.value
    output_path = _OUTPUT_PATH.value
    episode_length = _EPISODE_LENGTH.value
    render_every = _RENDER_EVERY.value
    num_episodes = _NUM_EPISODES.value
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    print(f"Environment: {env_name}")
    print(f"Output path: {output_path}")
    print(f"Episode length: {episode_length}")
    print(f"Render every: {render_every} steps")
    print(f"Num episodes: {num_episodes}")
    
    try:
        # Load environment
        env_cfg = registry.get_default_config(env_name)
        env = registry.load(env_name, config=env_cfg)
        print(f"Environment loaded successfully")
        
        # Get checkpoint path
        checkpoint_path = get_latest_checkpoint_path(checkpoint_dir)
        print(f"Checkpoint path: {checkpoint_path}")
        
        # Create inference function (this will load the checkpoint)
        inference_fn, params = load_checkpoint_and_create_inference_fn(checkpoint_path, env_name)
        print(f"Inference function created")
        
        # Get PPO parameters for reference
        ppo_params = get_rl_config(env_name)
        
        # Use the same environment setup as training to ensure compatibility
        # The policy was trained on a wrapped environment, so we need to use the same setup
        eval_env = wrapper.wrap_for_brax_training(
            env,
            episode_length=ppo_params.episode_length,
            action_repeat=ppo_params.action_repeat,
            randomization_fn=None,  # No randomization during inference
        )
        print(f"Using wrapped environment for evaluation (same as training)")
        
        # Render rollout using the wrapped environment
        frames, fps = render_rollout(eval_env, inference_fn, episode_length, render_every, num_episodes)
        
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