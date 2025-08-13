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
              ball_fell_off=1.0,
              velocity_penalty=1e-5,
              action_rate=0.0,
          ),
      ),
  )


class RelocateTouchSimple(leap_hand_base.LeapHandEnv):
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
    
    # Get object joint IDs (these are not controllable)
    self._obj_qids = mjx_env.get_qpos_ids(self.mj_model, ["OBJTx", "OBJTy", "OBJTz", "OBJRx", "OBJRy", "OBJRz"])
    
    # Get site IDs for palm and target
    self._palm_site_id = self._mj_model.site("S_grasp").id
    self._target_site_id = self._mj_model.site("target").id
    
    # Get object body ID
    self._obj_body_id = self._mj_model.body("Object").id
    
    # Initialize defaults from model qpos0 to match viewer
    self._qpos0 = jp.array(self._mj_model.qpos0)
    default_hand_pose = self._qpos0[self._hand_qids]
    self._default_pose = default_hand_pose
    
    # Get actuator limits for hand joints only
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Use exact XML initial pose (no randomization)
    q_hand = self._default_pose
    v_hand = jp.zeros_like(self._default_pose)

    # Start from model qpos0 so all non-hand joints match viewer exactly
    qpos = jp.array(self._mj_model.qpos0)
    qvel = jp.zeros_like(qpos)
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

    # State size = hand joints + previous actions
    state_size = len(self._hand_qids) + self.mjx_model.nu
    obs_history = jp.zeros(self._config.history_len * state_size)
    obs = self._get_obs(data, info, obs_history)
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = self._default_pose + action * self._config.action_scale
    # NOTE: no clipping.
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
    # Episode ends when object falls off table (z < -0.05) or when object is very close to target
    obj_pos = data.xpos[self._obj_body_id]
    target_pos = data.site_xpos[self._target_site_id]
    
    obj_to_target_dist = jp.linalg.norm(obj_pos - target_pos)
    obj_fell_off = obj_pos[2] < -0.05
    obj_reached_target = obj_to_target_dist < 0.05
    
    return obj_fell_off | obj_reached_target

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

    state = jp.concatenate([
        noisy_joint_angles,  # Hand joints (27 values: 6 base motion + 21 finger joints)
        info["last_act"],  # Previous actions (27 values)
    ])
    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    # Get palm, object, and target positions
    palm_pos = data.site_xpos[self._palm_site_id]
    obj_pos = data.xpos[self._obj_body_id]
    target_pos = data.site_xpos[self._target_site_id]
    
    # Get object orientation and velocities
    obj_quat = data.xquat[self._obj_body_id]
    obj_linvel = data.qvel[self._obj_qids[3:6]]  # Linear velocity
    obj_angvel = data.qvel[self._obj_qids[6:9]]  # Angular velocity
    
    # Get fingertip positions
    fingertip_positions = self.get_fingertip_positions(data)
    
    # Get joint torques
    joint_torques = data.actuator_force

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        joint_torques,
        fingertip_positions,
        palm_pos,
        obj_pos,
        target_pos,
        obj_quat,
        obj_linvel,
        obj_angvel,
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
    # Get positions
    palm_pos = data.site_xpos[self._palm_site_id]
    obj_pos = data.xpos[self._obj_body_id]
    target_pos = data.site_xpos[self._target_site_id]
    
    # Calculate distances
    palm_to_obj_dist = jp.linalg.norm(palm_pos - obj_pos)
    palm_to_target_dist = jp.linalg.norm(palm_pos - target_pos)
    obj_to_target_dist = jp.linalg.norm(obj_pos - target_pos)
    
    # Check if object is off table
    # Table surface is at z = 0.0, object radius is 0.04
    # Object is off table when z > table_surface + radius = 0.0 + 0.04 = 0.04
    obj_off_table = obj_pos[2] > 0.04
    
    # Check if object fell off table (for negative reward)
    obj_fell_off = obj_pos[2] < -0.05
    
    # Check if object is close to target
    # obj_close_to_target = obj_to_target_dist < 0.1
    obj_very_close_to_target = obj_to_target_dist < 0.05
    
    # Get velocities for penalty
    qvel = data.qvel
    
    rewards = {
        "get_to_ball": -palm_to_obj_dist,  # Take hand to object
        "ball_off_table": jp.where(obj_off_table, 1.0, 0.0),  # Bonus for lifting object
        "make_hand_go_to_target": jp.where(obj_off_table, -palm_to_target_dist, 0.0),  # Make hand go to target when object is lifted
        "make_ball_go_to_target": jp.where(obj_off_table, -obj_to_target_dist, 0.0),  # Make object go to target when lifted
        # "ball_close_to_target": jp.where(obj_close_to_target, 1.0, 0.0),  # Bonus for object close to target
        "ball_very_close_to_target": jp.where(obj_very_close_to_target, 1.0, 0.0),  # Bonus for object very close to target
        "ball_fell_off": jp.where(obj_fell_off, -10.0, 0.0),  # Negative reward for falling off table
        "velocity_penalty": jp.sum(jp.square(qvel)),  # Penalty for high velocities
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
    }
    
    return rewards

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Domain randomization for relocate environment."""
  # Randomize object and target positions by ±0.5
  rng, obj_rng, target_rng = jax.random.split(rng, 3)
  
  # Randomize object X and Y positions
  obj_x = jax.random.uniform(obj_rng, (), minval=-0.5, maxval=0.5)
  obj_y = jax.random.uniform(obj_rng, (), minval=-0.5, maxval=0.5)
  
  # Randomize target X and Y positions  
  target_x = jax.random.uniform(target_rng, (), minval=-0.5, maxval=0.5)
  target_y = jax.random.uniform(target_rng, (), minval=-0.5, maxval=0.5)
  
  # Update object initial position
  obj_qids = mjx_env.get_qpos_ids(model, ["OBJTx", "OBJTy"])
  qpos0 = model.qpos0.at[obj_qids[0]].set(obj_x)
  qpos0 = qpos0.at[obj_qids[1]].set(obj_y)
  
  # Update target site position
  target_site_id = model.site("target").id
  site_pos = model.site_pos.at[target_site_id].set(jp.array([target_x, target_y, 0.2]))
  
  in_axes = jax.tree_util.tree_map(lambda _: None, model)
  in_axes = in_axes.tree_replace({
      "qpos0": 0,
      "site_pos": 0,
  })

  model = model.tree_replace({
      "qpos0": qpos0,
      "site_pos": site_pos,
  })

  return model, in_axes
