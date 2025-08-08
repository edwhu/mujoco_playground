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
"""Rotate-z with leap hand."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

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
              angvel=1.0,
              linvel=0.0,
              pose=0.0,
              torques=0.0,
              energy=0.0,
              termination=-100.0,
              action_rate=0.0,
          ),
      ),
  )


class AllosaurusRotateZAxis(leap_hand_base.LeapHandEnv):
  """Rotate a allosaurus around the z-axis as fast as possible wihout dropping it."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.ALLOSAURUS_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    self._object_qids = mjx_env.get_qpos_ids(self.mj_model, ["allosaurus_freejoint"])
    self._floor_geom_id = self._mj_model.geom("floor").id

    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos)
    self._default_pose = self._init_q[self._hand_qids]
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Randomize hand qpos and qvel.
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    q_hand = jp.clip(
        self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
        self._lowers,
        self._uppers,
    )
    v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

    # Randomize allosaurus qpos and qvel.
    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    start_pos = jp.array([0.1, 0.0, 0.05]) + jax.random.uniform(
        p_rng, (3,), minval=-0.01, maxval=0.01
    )
    start_quat = leap_hand_base.uniform_quat(quat_rng)
    q_allosaurus = jp.array([*start_pos, *start_quat])
    v_allosaurus = jp.zeros(6)

    qpos = jp.concatenate([q_hand, q_allosaurus])
    qvel = jp.concatenate([v_hand, v_allosaurus])
    data = mjx_env.init(
        self.mjx_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=q_hand,
        mocap_pos=jp.array([-100, -100, -100]),  # Hide goal for this task.
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "last_allosaurus_angvel": jp.zeros(3),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs_history = jp.zeros(self._config.history_len * 32)
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

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_allosaurus_angvel"] = self.get_object_angvel(data, "allosaurus")
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_object_position(data, "allosaurus")[2] < -0.05
    return fall_termination

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
        noisy_joint_angles,  # 16
        info["last_act"],  # 16
    ])  # 48
    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    allosaurus_pos = self.get_object_position(data, "allosaurus")
    palm_pos = self.get_palm_position(data)
    allosaurus_pos_error = palm_pos - allosaurus_pos
    allosaurus_quat = self.get_object_orientation(data, "allosaurus")
    allosaurus_angvel = self.get_object_angvel(data, "allosaurus")
    allosaurus_linvel = self.get_object_linvel(data, "allosaurus")
    fingertip_positions = self.get_fingertip_positions(data)
    joint_torques = data.actuator_force

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        joint_torques,
        fingertip_positions,
        allosaurus_pos_error,
        allosaurus_quat,
        allosaurus_angvel,
        allosaurus_linvel,
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
    del metrics  # Unused.
    allosaurus_pos = self.get_object_position(data, "allosaurus")
    palm_pos = self.get_palm_position(data)
    allosaurus_pos_error = palm_pos - allosaurus_pos
    allosaurus_angvel = self.get_object_angvel(data, "allosaurus")
    allosaurus_linvel = self.get_object_linvel(data, "allosaurus")
    return {
        "angvel": self._reward_angvel(allosaurus_angvel, allosaurus_pos_error),
        "linvel": self._cost_linvel(allosaurus_linvel),
        "termination": done,
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "pose": self._cost_pose(data.qpos[self._hand_qids]),
        "torques": self._cost_torques(data.actuator_force),
        "energy": self._cost_energy(
            data.qvel[self._hand_dqids], data.actuator_force
        ),
    }

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_linvel(self, allosaurus_linvel: jax.Array) -> jax.Array:
    return jp.linalg.norm(allosaurus_linvel, ord=1, axis=-1)

  def _reward_angvel(
      self, allosaurus_angvel: jax.Array, allosaurus_pos_error: jax.Array
  ) -> jax.Array:
    # Unconditionally maximize angvel in the z-direction.
    del allosaurus_pos_error  # Unused.
    return allosaurus_angvel @ jp.array([0.0, 0.0, 1.0])

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
    return jp.sum(jp.square(joint_angles - self._default_pose))