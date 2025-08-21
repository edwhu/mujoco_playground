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
import numpy as np
from orbax import checkpoint as ocp
from absl import app
from absl import flags
from collections import defaultdict
from copy import deepcopy

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
    "env_name", "LeapDoorOpenRandom", "Environment name"
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
_ENABLE_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "enable_domain_randomization", False, "Enable domain randomization during rendering"
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
    
    # RNG
    rng = jax.random.PRNGKey(123)
    
    # For evaluation we use parallel environments
    print(f"Using {num_episodes} parallel environments for evaluation")
    
    # Use the environment that was already created in main()
    eval_env = env
    
    # JIT compile functions for the environment
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Reset all parallel episodes at once
    rng, reset_rng = jax.random.split(rng)
    reset_rng = jp.asarray(jax.random.split(reset_rng, num_episodes))
    state = jit_reset(reset_rng)
    
    # Debug: Check actual door positions in the initial state after reset
    print(f"Starting rendering for {num_episodes} episode(s), episode length: {episode_length}")
    # print("Checking actual door positions in initial state:")
    # if hasattr(state, 'data') and hasattr(state.data, 'qpos'):
    #     # Check door hinge position (should be 0 initially, but frame position affects door)
    #     door_qid = 4  # Assuming door hinge is at index 4, adjust if needed
    #     door_positions = state.data.qpos[:, door_qid] if hasattr(state.data.qpos, 'shape') and len(state.data.qpos.shape) > 1 else state.data.qpos[door_qid]
    #     print(f"   Door hinge positions: {door_positions}")
    
    # # Also check if we can access the actual randomized model being used
    # if hasattr(eval_env, '_mjx_model_v') and hasattr(eval_env, '_frame_body_id') and eval_env._frame_body_id is not None:
    #     frame_id = eval_env._frame_body_id
    #     actual_door_positions = eval_env._mjx_model_v.body_pos[:, frame_id]
    #     print(f"   Actual frame body positions from randomized model: {actual_door_positions}")
    #     print(f"   Unique frame positions: {len(jp.unique(actual_door_positions, axis=0))}")
    
    # Track parallel trajectories and episode termination
    parallel_trajectories = [[] for _ in range(num_episodes)]
    episode_terminated = [False] * num_episodes
    episode_lengths = [0] * num_episodes
    
    # Add initial states
    for i in range(num_episodes):
        parallel_trajectories[i].append(state)
        episode_lengths[i] += 1

    for step in range(episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        
        # Add states to trajectories and check termination
        for i in range(num_episodes):
            if not episode_terminated[i]:
                parallel_trajectories[i].append(state)
                episode_lengths[i] += 1
                
                # Check if this episode terminated
                if hasattr(state.done, 'shape') and len(state.done.shape) > 0:
                    if state.done[i]:
                        episode_terminated[i] = True
                        print(f"Episode {i+1} terminated at step {step + 1}")
                else:
                    if state.done:
                        episode_terminated[i] = True
                        print(f"Episode {i+1} terminated at step {step + 1}")
        
        # Logging
        if hasattr(state.reward, 'shape') and len(state.reward.shape) > 0:
            reward_mean = jp.mean(state.reward).item()
            done_mean = jp.mean(state.done).item()
            print(f"Step {step + 1}: reward={reward_mean:.4f}, done={done_mean:.4f}")
        else:
            print(f"Step {step + 1}: reward={state.reward:.4f}, done={state.done}")
        
        # Stop early if all episodes terminate
        if all(episode_terminated):
            print(f"All episodes terminated at step {step + 1}")
            break
    
    print(f"Parallel rollout completed. Episode lengths: {episode_lengths}")
    
    # Convert parallel trajectories to sequential for rendering
    print("Converting parallel trajectories to sequential for rendering...")
    # sequential_trajectory = []
    episode_trajectories = defaultdict(list) # list of episode trajectories
    episode_indices = []  # Track which episode each state belongs to
    
    for ep in range(num_episodes):
        episode_trajectory = parallel_trajectories[ep]
        # Only take the actual episode length (not the full trajectory)
        episode_trajectory = episode_trajectory[:episode_lengths[ep]]
        
        # Extract the individual episode state from the parallel batch
        for state in episode_trajectory:
            # Extract the state for this specific episode from the batch
            single_state = jax.tree.map(lambda x: x[ep] if hasattr(x, 'shape') and len(x.shape) > 0 else x, state)
            episode_trajectories[ep].append(single_state)
    
    print(f"Parallel rollout completed. Episode lengths: {episode_lengths}")

    total_frames = sum([len(t) for t in episode_trajectories.values()])
    print(f"All trajectories have {total_frames} total states")
    
    # Render the rollout
    print("Rendering video...")
    fps = 1.0 / env.dt / render_every
    print(f"FPS: {fps}")
    
    frames = [] # list of frames for all episodes
        
    # Sample frames for rendering
    for episode_num, episode_trajectory in episode_trajectories.items():
        traj = episode_trajectory[::render_every]
        print(f"Rendering {len(traj)} frames for episode {episode_num+1}")
        
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



        # Sync model for this episode if domain randomization is enabled
        if hasattr(env, '_mjx_model_v') and hasattr(env, '_in_axes'):
            print(f"Syncing model for episode {episode_num+1} with randomized body positions")
            
            # Get the randomized model for this episode
            episode_model = env._mjx_model_v
            episode_in_axes = env._in_axes
            
            # Update the rendering model with all randomized body positions for this episode
            mj_model = env.mj_model
            
            # Store original positions for all bodies that might be randomized
            original_positions = {}
            
            # Check which body positions are randomized (have in_axes != None)
            if hasattr(episode_in_axes, 'body_pos') and episode_in_axes.body_pos is not None:
                # Get all randomized body positions for this episode
                episode_body_pos = episode_model.body_pos[episode_num]
                original_body_pos = np.array(mj_model.body_pos)
                
                # Update the rendering model with this episode's body positions
                mj_model.body_pos[:] = episode_body_pos
                
                try:
                    episode_frames = env.render(
                        traj, height=480, width=640, scene_option=scene_option, camera="fixed"
                    )
                    frames.extend(episode_frames)
                finally:
                    # Restore original body positions
                    mj_model.body_pos[:] = original_body_pos
            else:
                # No body position randomization, render normally
                episode_frames = env.render(
                    traj, height=480, width=640, scene_option=scene_option, camera="fixed"
                )
                frames.extend(episode_frames)
        else:
            print("Using standard render (no domain randomization)")
            episode_frames = env.render(
                traj, height=480, width=640, scene_option=scene_option, camera="fixed"
            )
            frames.extend(episode_frames)
    
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
    enable_domain_randomization = _ENABLE_DOMAIN_RANDOMIZATION.value
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    print(f"Environment: {env_name}")
    print(f"Output path: {output_path}")
    print(f"Episode length: {episode_length}")
    print(f"Render every: {render_every} steps")
    print(f"Num episodes: {num_episodes}")
    print(f"Domain randomization: {enable_domain_randomization}")
    
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
        
        # Get domain randomization function using the same approach as training
        randomization_fn = None
        if enable_domain_randomization:
            base_randomization_fn = registry.get_domain_randomizer(env_name)
            if base_randomization_fn is not None:
                # Create a wrapper function that matches the expected signature
                # The wrapper expects: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
                # But the registry function is: Callable[[mjx.Model, rng], Tuple[mjx.Model, mjx.Model]]
                def create_randomization_wrapper(base_fn, num_episodes):
                    # Create RNG keys for each parallel episode
                    import time
                    import random
                    seed = random.randint(0, 1000000000)
                    key_env = jax.random.PRNGKey(seed)
                    randomization_rng = jax.random.split(key_env, num_episodes)
                    
                    def wrapper_fn(model):
                        # Add error handling and numerical stability checks
                        try:
                            result, in_axes = base_fn(model, randomization_rng)
                            # Check for numerical issues
                            if hasattr(result, 'body_pos'):
                                # Ensure body positions are finite
                                if not jp.all(jp.isfinite(result.body_pos)):
                                    print("⚠️  Warning: Non-finite body positions detected, using original model")
                                    return model, jax.tree_util.tree_map(lambda _: None, model)
                            return result, in_axes
                        except Exception as e:
                            print(f"⚠️  Warning: Domain randomization failed: {e}")
                            print("   Falling back to original model without randomization")
                            return model, jax.tree_util.tree_map(lambda _: None, model)
                    
                    return wrapper_fn, seed
                
                randomization_fn, seed = create_randomization_wrapper(base_randomization_fn, num_episodes)
                print(f"✅ Using domain randomization for {env_name} with {num_episodes} parallel episodes (seed: {seed})")
            else:
                print(f"⚠️  No domain randomization function found for {env_name}")
        else:
            print("ℹ️  Domain randomization disabled")
        
        # Use the same environment setup as training to ensure compatibility
        # The policy was trained on a wrapped environment, so we need to use the same setup
        try:
            eval_env = wrapper.wrap_for_brax_training(
                env,
                episode_length=ppo_params.episode_length,
                action_repeat=ppo_params.action_repeat,
                randomization_fn=randomization_fn,  # Enable randomization if available
            )
            print(f"Using wrapped environment for evaluation with randomization: {randomization_fn is not None}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to create wrapped environment with randomization: {e}")
            print("   Falling back to environment without domain randomization")
            eval_env = wrapper.wrap_for_brax_training(
                env,
                episode_length=ppo_params.episode_length,
                action_repeat=ppo_params.action_repeat,
                randomization_fn=None,  # Disable randomization
            )
            print("Using wrapped environment for evaluation without randomization")
        
        # Render rollout using the wrapped environment
        frames, fps = render_rollout(
            eval_env, 
            inference_fn, 
            episode_length, 
            render_every, 
            num_episodes
        )
        
        # Save video
        media.write_video(output_path, frames, fps=fps)
        print(f"Video saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        if "cuSolver internal error" in str(e) or "overflow encountered" in str(e):
            print("\n💡 Suggestion: This error might be caused by:")
            print("   1. Too many parallel episodes causing GPU memory issues")
            print("   2. Domain randomization creating numerical instability")
            print("   Try reducing --num_episodes or disabling --enable_domain_randomization")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    app.run(main) 