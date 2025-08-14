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
              get_to_ball=0.1,
              ball_off_table=1.0,
              make_hand_go_to_target=0.5,
              make_ball_go_to_target=0.5,
              ball_close_to_target=10.0,
              ball_very_close_to_target=20.0,
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
        xml_path="mujoco_playground/_src/manipulation/leap_hand/xmls/scene_mjx_relocate.xml",
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
    self._obj_body = self._mj_model.body("Object").id
    self._obj_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body("Object").jntadr[0]
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

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Randomize object position like in pick_cartesian.py
    rng, rng_obj_x, rng_obj_y = jax.random.split(rng, 3)
    obj_range = 0.3  # Similar to box_init_range in pick_cartesian.py
    obj_pos = jp.array([
        jax.random.uniform(rng_obj_x, (), minval=-obj_range, maxval=obj_range),  # Randomize X position
        jax.random.uniform(rng_obj_y, (), minval=-obj_range, maxval=obj_range),  # Randomize Y position
        0.04,  # Fixed Z position on table
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
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    # State size = hand joints + previous actions + target position (x, y only)
    state_size = len(self._hand_qids) + self.mjx_model.nu + 2
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
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name

    # Check for NaN in final reward and end episode if detected
    has_nan_reward = jp.isnan(reward)
    reward = jp.where(has_nan_reward, jp.zeros(()), reward)
    done = jp.where(has_nan_reward, jp.ones(()), done)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Episode ends only when object falls off table (z < -0.05)
    # No early termination when close to target - let it run full episode
    obj_pos = data.xpos[self._obj_body]
    obj_fell_off = obj_pos[2] < -0.05
    
    return obj_fell_off

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

    # Get target position for the policy to know where to move the object
    target_pos = data.site_xpos[self._target_site_id]
    target_xy = target_pos[:2]  # Only x and y coordinates since z is constant
    
    state = jp.concatenate([
        noisy_joint_angles,  # Hand joints (27 values: 6 base motion + 21 finger joints)
        info["last_act"],  # Previous actions (27 values)
        target_xy,  # Target position (2 values: x, y)
    ])
    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    # Get palm and target positions
    palm_pos = data.site_xpos[self._palm_site_id]
    target_pos = data.site_xpos[self._target_site_id]
    
    # Get object data using sensors (only what's needed for reward and future state prediction)
    obj_pos = data.xpos[self._obj_body]  # Use body position directly
    obj_linvel = data.qvel[self._obj_qposadr:self._obj_qposadr+3]  # Linear velocity (x,y,z)
    obj_angvel = data.qvel[self._obj_qposadr+3:self._obj_qposadr+6]  # Angular velocity (rx,ry,rz)
    
    # Get fingertip positions (useful for understanding hand configuration)
    fingertip_positions = self.get_fingertip_positions(data)

    privileged_state = jp.concatenate([
        state,
        joint_angles,  # Hand pose affects future actions
        data.qvel[self._hand_dqids],  # Hand velocities affect future positions
        fingertip_positions,  # Hand configuration affects grasping
        palm_pos,
        obj_pos,
        target_pos,
        obj_linvel,  # Object velocity helps predict future position
        obj_angvel,  # Object angular velocity affects future orientation
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
    # Get positions using sensors
    palm_pos = data.site_xpos[self._palm_site_id]
    obj_pos = data.xpos[self._obj_body]  # Use body position directly
    target_pos = data.site_xpos[self._target_site_id]
    
    # Calculate distances
    palm_to_obj_dist = jp.linalg.norm(palm_pos - obj_pos)
    palm_to_target_dist = jp.linalg.norm(palm_pos - target_pos)
    obj_to_target_dist = jp.linalg.norm(obj_pos - target_pos)
    
    # Check if object is off table (lifted)
    obj_off_table = obj_pos[2] > 0.05
    
    # Check if object is close to target
    obj_close_to_target = obj_to_target_dist < 0.1
    obj_very_close_to_target = obj_to_target_dist < 0.05
    
    # Follow Adroit reward structure as dictionary components
    rewards = {
        "get_to_ball": -palm_to_obj_dist,  # Take hand to object (negative distance)
        "ball_off_table": jp.where(obj_off_table, 1.0, 0.0),  # Bonus for lifting the object
        "make_hand_go_to_target": jp.where(obj_off_table, -palm_to_target_dist, 0.0),  # Make hand go to target when lifted
        "make_ball_go_to_target": jp.where(obj_off_table, -obj_to_target_dist, 0.0),  # Make object go to target when lifted
        "ball_close_to_target": jp.where(obj_close_to_target, 1.0, 0.0),  # Bonus for object close to target
        "ball_very_close_to_target": jp.where(obj_very_close_to_target, 1.0, 0.0),  # Bonus for object very close to target
    }
    
    return rewards

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Domain randomization for relocate environment."""
  # No domain randomization - object randomization happens in reset()
  return model, None
