# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Object relocation with leap hand."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.leap_hand import base as leap_hand_base
from mujoco_playground._src.manipulation.leap_hand import leap_hand_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      action_scale=0.6,
      action_repeat=1,
      episode_length=500,
      early_termination=True,
      history_len=1,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              fingertips_to_object=1.0,  # Reward for finger tips getting closer to object
              cube_height=5.0,  # Reward for lifting the cube
              cube_lifted=500.0,  # Large reward for successfully lifting cube above threshold
          ),
      ),
  )


class Relocate(leap_hand_base.LeapHandEnv):
  """Relocate an object to a target position using the Leap Hand."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path="mujoco_playground/_src/manipulation/leap_hand/xmls/relocate_scene.xml",
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    # Get hand joint IDs (including base motion joints) - only hand joints are controllable
    hand_joint_names = ["H_Tx", "H_Ty", "H_Tz", "H_Rx", "H_Ry", "H_Rz"] + consts.JOINT_NAMES
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, hand_joint_names)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, hand_joint_names)
    
    # Get object body ID and qpos address (following the pattern from pick_cartesian.py)
    self._obj_body = self._mj_model.body("cube").id
    self._obj_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body("cube").jntadr[0]
    ]
    
    # Get site IDs for palm and target
    self._palm_site_id = self._mj_model.site("palm_site").id
    self._target_site_id = self._mj_model.site("target").id
    
    # Initialize defaults from model qpos0 to match viewer
    self._qpos0 = jp.array(self._mj_model.qpos0)
    default_hand_pose = self._qpos0[self._hand_qids]
    self._default_pose = default_hand_pose
    
    # Get actuator limits for hand joints only
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T
    
    # Precompute hand body IDs for contact detection
    self._hand_body_ids = self._get_hand_body_ids()

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Randomize object position like in pick_cartesian.py
    rng, rng_obj_x, rng_obj_y = jax.random.split(rng, 3)
    obj_range = 0.3  # Similar to box_init_range in pick_cartesian.py
    obj_pos = jp.array([
        jax.random.uniform(rng_obj_x, (), minval=-obj_range, maxval=obj_range),  # Randomize X position
        jax.random.uniform(rng_obj_y, (), minval=-obj_range, maxval=obj_range),  # Randomize Y position
        -0.215,  # Fixed Z position on table
    ])

    # Use exact XML initial pose for hand (no randomization)
    q_hand = self._default_pose
    v_hand = jp.zeros_like(self._default_pose)

    # Start from model qpos0 so all non-hand joints match viewer exactly
    qpos = jp.array(self._mj_model.qpos0)
    qvel = jp.zeros(self._mj_model.nv)  # Use correct velocity size
    
    # Set object position (following pick_cartesian.py pattern)
    qpos = qpos.at[self._obj_qposadr:self._obj_qposadr + 3].set(obj_pos)
    
    # Set hand joints
    qpos = qpos.at[self._hand_qids].set(q_hand)
    qvel = qvel.at[self._hand_dqids].set(v_hand)

    data = mjx_env.init(
        self.mjx_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=q_hand,
        # No mocap bodies in this environment, so don't pass mocap_pos
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "last_obj_height": obj_pos[2],  # Store initial object height
        "last_fingertip_distances": jp.zeros(4),  # Store initial fingertip distances
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    # State size = hand joints + previous actions + target position (x, y only) + DEBUG cube state
    state_size = len(self._hand_qids) + self.mjx_model.nu + 2 + 13  # +13 for obj_pos(3) + obj_quat(4) + obj_linvel(3) + obj_angvel(3)
    obs_history = jp.zeros(self._config.history_len * state_size)
    obs = self._get_obs(data, info, obs_history)
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = self._default_pose + action
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    obs = self._get_obs(data, state.info, state.obs["state"])
    done = jp.array(False)  # Only terminate when cube is lifted (handled in reward logic)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name

    # Check for cube_lifted condition and end episode if true
    cube_lifted_done = rewards["cube_lifted"] > 0.0
    done = done | cube_lifted_done

    # Check for NaN in final reward and end episode if detected
    has_nan_reward = jp.isnan(reward)
    reward = jp.where(has_nan_reward, jp.zeros(()), reward)
    done = jp.where(has_nan_reward, jp.ones(()), done)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    # Update last object height and fingertip distances for next step
    obj_pos = data.xpos[self._obj_body]
    state.info["last_obj_height"] = obj_pos[2]
    
    # Calculate current fingertip distances for next step
    fingertip_positions = self.get_fingertip_positions(data)
    current_fingertip_distances = jp.array([
        jp.linalg.norm(fingertip_positions[0] - obj_pos),  # Index finger
        jp.linalg.norm(fingertip_positions[1] - obj_pos),  # Middle finger  
        jp.linalg.norm(fingertip_positions[2] - obj_pos),  # Ring finger
        jp.linalg.norm(fingertip_positions[3] - obj_pos),  # Thumb
    ])
    state.info["last_fingertip_distances"] = current_fingertip_distances
    
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state



  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array
  ) -> Dict[str, jax.Array]:
    joint_angles = data.qpos[self._hand_qids]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )
    
    # Get object data using sensors (only what's needed for reward and future state prediction)
    obj_pos = self.get_cube_position(data)
    obj_quat = self.get_cube_orientation(data)
    obj_linvel = self.get_cube_linvel(data)
    obj_angvel = self.get_cube_angvel(data)
    
    # Add noise to cube state for policy observations (following reorient.py pattern)
    info["rng"], pos_rng, quat_rng, linvel_rng, angvel_rng = jax.random.split(info["rng"], 5)
    
    # Add noise to cube position
    noisy_obj_pos = (
        obj_pos
        + (2 * jax.random.uniform(pos_rng, shape=obj_pos.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos  # Using joint_pos scale for position noise
    )
    
    # Add noise to cube quaternion (normalize to keep it valid)
    noisy_obj_quat = mjx._src.math.normalize(
        obj_quat
        + jax.random.normal(quat_rng, shape=(4,))
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos  # Using joint_pos scale for quaternion noise
    )
    
    # Add noise to velocities
    noisy_obj_linvel = (
        obj_linvel
        + (2 * jax.random.uniform(linvel_rng, shape=obj_linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )
    
    noisy_obj_angvel = (
        obj_angvel
        + (2 * jax.random.uniform(angvel_rng, shape=obj_angvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )
    
    state = jp.concatenate([
        noisy_joint_angles,  # Hand joints (27 values: 6 base motion + 21 finger joints)
        info["last_act"],  # Previous actions (27 values)
        # DEBUG: Add cube state information (noisy for policy)
        noisy_obj_pos,  # Object position (3 values: x, y, z)
        noisy_obj_quat,  # Object quaternion (4 values: qw, qx, qy, qz)
        noisy_obj_linvel,  # Object linear velocity (3 values: vx, vy, vz)
        noisy_obj_angvel,  # Object angular velocity (3 values: ωx, ωy, ωz)
    ])
    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    # Get palm and target positions
    palm_pos = data.site_xpos[self._palm_site_id]
    
    # Get world-frame fingertip positions for critic
    fingertip_site_names = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
    world_fingertip_positions = jp.concatenate([
        data.site_xpos[self._mj_model.site(name).id] for name in fingertip_site_names
    ], axis=-1)

    privileged_state = jp.concatenate([
        state,
        joint_angles,  # Hand pose affects future actions
        data.qvel[self._hand_dqids],  # Hand velocities affect future positions
        world_fingertip_positions,  # World-frame hand configuration affects grasping
        palm_pos,
        obj_pos,  # True object position for critic
        obj_linvel,  # True object velocity for critic
        obj_angvel,  # True object angular velocity for critic
        data.qvel,  # For velocity penalty
    ])

    return {
        "state": obs_history,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    # Get object position directly from data
    obj_pos = data.xpos[self._obj_body]
    # Adjust z position to encourage gripping at the bottom of the object
    obj_pos = obj_pos.at[2].set(obj_pos[2] - 0.035)
    
    # Get world-frame fingertip positions directly from data using sites
    fingertip_site_names = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
    world_fingertip_positions = jp.array([
        data.site_xpos[self._mj_model.site(name).id] for name in fingertip_site_names
    ])
    
    # Calculate distances using world-frame positions
    current_fingertip_distances = jp.array([
        jp.linalg.norm(world_fingertip_positions[0] - obj_pos),  # Index finger
        jp.linalg.norm(world_fingertip_positions[1] - obj_pos),  # Middle finger  
        jp.linalg.norm(world_fingertip_positions[2] - obj_pos),  # Ring finger
        jp.linalg.norm(world_fingertip_positions[3] - obj_pos),  # Thumb
    ])
        
    # Get last timestep's distances
    last_fingertip_distances = info["last_fingertip_distances"]
    
    # Calculate distance improvements
    fingertip_distance_improvements = last_fingertip_distances - current_fingertip_distances
    
    # Sum the improvements across all fingertips
    fingertips_to_object_reward = jp.sum(fingertip_distance_improvements)
    
    # Check if any fingertip is touching the object (using contact detection)
    hand_obj_contact = self._check_hand_object_contact(obj_pos, world_fingertip_positions)
    
    # Calculate cube height change
    current_obj_height = obj_pos[2]
    last_obj_height = info["last_obj_height"]
    height_change = current_obj_height - last_obj_height
    
    # Only reward height change if touching the object
    cube_height_reward = jp.where(hand_obj_contact, height_change, 0.0)
    
    # Check if cube is lifted above threshold (0.25)
    cube_lifted = current_obj_height > 0.1
    cube_lifted_reward = jp.where(cube_lifted & hand_obj_contact, 1.0, 0.0)
    
    rewards = {
        "fingertips_to_object": fingertips_to_object_reward,
        "cube_height": cube_height_reward,
        "cube_lifted": cube_lifted_reward,
    }
    
    return rewards

  def _get_hand_body_ids(self) -> list[int]:
    """Get hand body IDs for contact detection."""
    hand_body_ids = []
    for body_name in ["palm", "if_bs", "if_px", "if_md", "if_ds", "mf_bs", "mf_px", "mf_md", "mf_ds", "rf_bs", "rf_px", "rf_md", "rf_ds", "th_mp", "th_bs", "th_px", "th_ds"]:
      try:
        hand_body_ids.append(self._mj_model.body(body_name).id)
      except:
        pass  # Skip if body doesn't exist
    return hand_body_ids

  def _check_hand_object_contact(self, obj_pos: jax.Array, fingertip_positions: jax.Array) -> jax.Array:
    """Check if hand is in contact with the object using distance-based detection."""
    # Calculate distances from all fingertips to object
    distances = jp.array([
        jp.linalg.norm(fingertip_positions[0] - obj_pos),  # Index finger
        jp.linalg.norm(fingertip_positions[1] - obj_pos),  # Middle finger  
        jp.linalg.norm(fingertip_positions[2] - obj_pos),  # Ring finger
        jp.linalg.norm(fingertip_positions[3] - obj_pos),  # Thumb
    ])
    
    # Check if any fingertip is within contact threshold (0.06 meters)
    contact_threshold = 0.06
    has_contact = jp.any(distances < contact_threshold)
    
    return has_contact

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Domain randomization for relocate environment."""
  # No domain randomization - object randomization happens in reset()
  return model, None
