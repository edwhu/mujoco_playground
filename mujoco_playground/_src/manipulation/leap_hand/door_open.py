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
"""Door opening with leap hand."""

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
              get_to_handle=0.1,
              velocity_penalty=1e-5,
              action_rate=0.0,
              door_angle=1.0,
              door_open=100.0,
          ),
      ),
      # Door randomization ranges (delta added to nominal frame body pose)
      door_randomization=config_dict.create(
          enabled=True,
          dx=0.15,  # ± in meters (lateral)
          dy=0.2,  # ± in meters (toward/away)
          dz=0.05,  # ± in meters (height)
          yaw_deg=20.0,   # ± degrees about z
          pitch_deg=8.0,  # ± degrees about y
          roll_deg=8.0,   # ± degrees about x
          hinge_deg=5.0,  # initial hinge angle jitter (± degrees)
          min_z=0.20,     # clamp door base z to stay above table
      ),
  )


class DoorOpen(leap_hand_base.LeapHandEnv):
  """Open a door using the Leap Hand."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path="mujoco_playground/_src/manipulation/leap_hand/xmls/scene_mjx_door.xml",
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    # Get hand joint IDs (including base motion joints) - only hand joints are controllable
    hand_joint_names = ["H_Tx", "H_Rx", "H_Ry", "H_Rz"] + consts.JOINT_NAMES
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, hand_joint_names)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, hand_joint_names)

    # Get door and latch joint IDs (these are not controllable)
    self._door_qid = mjx_env.get_qpos_ids(self.mj_model, ["door_hinge"])[0]
    self._latch_qid = mjx_env.get_qpos_ids(self.mj_model, ["latch"])[0]

    # Get site IDs for palm and handle
    self._palm_site_id = self._mj_model.site("grasp_site").id
    self._handle_site_id = self._mj_model.site("S_handle").id

    # Cache the body id of the door frame so we can randomize its position
    self._frame_body_id = self._mj_model.body("frame").id
    # Nominal frame position from model (XYZ)
    self._frame_pos0 = jp.array(self.mjx_model.body_pos[self._frame_body_id])
    # Nominal frame orientation (w, x, y, z)
    self._frame_quat0 = jp.array(self.mjx_model.body_quat[self._frame_body_id])

    # Initialize defaults from model qpos0 to match viewer
    self._qpos0 = jp.array(self._mj_model.qpos0)
    default_hand_pose = self._qpos0[self._hand_qids]
    self._default_pose = default_hand_pose

    # Get actuator limits for hand joints only
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T

  def _randomize_door_pose(self, rng: jax.Array) -> jax.Array:
    """Randomize the 'frame' body pose (position + orientation) within ranges.

    Returns the new RNG.
    """
    if not self._config.door_randomization.enabled:
      return rng
    rng, key_pos = jax.random.split(rng)
    # Position deltas in [-range, range]
    ranges = jp.array([
        self._config.door_randomization.dx,
        self._config.door_randomization.dy,
        self._config.door_randomization.dz,
    ])
    deltas = (2.0 * jax.random.uniform(key_pos, (3,)) - 1.0) * ranges
    new_pos = self._frame_pos0 + deltas
    new_pos = new_pos.at[2].set(jp.maximum(new_pos[2], self._config.door_randomization.min_z))

    # Orientation deltas (degrees → radians)
    rng, key_ang = jax.random.split(rng)
    ang_scales = jp.array([
        self._config.door_randomization.roll_deg,   # about x
        self._config.door_randomization.pitch_deg,  # about y
        self._config.door_randomization.yaw_deg,    # about z
    ]) * (jp.pi / 180.0)
    angs = (2.0 * jax.random.uniform(key_ang, (3,)) - 1.0) * ang_scales
    roll, pitch, yaw = angs[0], angs[1], angs[2]

    def quat_axis_angle(axis, angle):
      half = angle * 0.5
      s = jp.sin(half)
      return jp.array([jp.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])

    def quat_mul(q1, q2):
      w1,x1,y1,z1 = q1
      w2,x2,y2,z2 = q2
      return jp.array([
          w1*w2 - x1*x2 - y1*y2 - z1*z2,
          w1*x2 + x1*w2 + y1*z2 - z1*y2,
          w1*y2 - x1*z2 + y1*w2 + z1*x2,
          w1*z2 + x1*y2 - y1*x2 + z1*w2,
      ])

    # Z-Y-X (yaw, pitch, roll)
    qz = quat_axis_angle(jp.array([0.0, 0.0, 1.0]), yaw)
    qy = quat_axis_angle(jp.array([0.0, 1.0, 0.0]), pitch)
    qx = quat_axis_angle(jp.array([1.0, 0.0, 0.0]), roll)
    q_delta = quat_mul(qz, quat_mul(qy, qx))
    new_quat = quat_mul(q_delta, self._frame_quat0)

    # Apply to mjx model (immutable tree)
    body_pos = self.mjx_model.body_pos.at[self._frame_body_id].set(new_pos)
    body_quat = self.mjx_model.body_quat.at[self._frame_body_id].set(new_quat)
    self.mjx_model = self.mjx_model.tree_replace({"body_pos": body_pos, "body_quat": body_quat})
    return rng

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Per-episode door placement randomization (before creating data)
    rng = self._randomize_door_pose(rng)

    # Use exact XML initial pose (no randomization) for hand
    q_hand = self._default_pose
    v_hand = jp.zeros_like(self._default_pose)

    # Start from model qpos0 so all non-hand joints match viewer exactly
    qpos = jp.array(self._mj_model.qpos0)
    qvel = jp.zeros_like(qpos)
    # Set hand joints
    qpos = qpos.at[self._hand_qids].set(q_hand)
    qvel = qvel.at[self._hand_dqids].set(v_hand)
    # Randomize door hinge small angle around closed
    rng, key_hinge = jax.random.split(rng)
    hinge = (2.0 * jax.random.uniform(key_hinge) - 1.0) * (self._config.door_randomization.hinge_deg * jp.pi / 180.0)
    qpos = qpos.at[self._door_qid].set(hinge)
    # Latch closed at start
    qpos = qpos.at[self._latch_qid].set(0.0)

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
        "last_door_angle": jp.zeros(1),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    # State size is 20 hand joints + 20 previous actions = 40
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

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_door_angle"] = data.qpos[self._door_qid:self._door_qid+1]
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Episode ends when door is sufficiently open (~90° with small tolerance)
    angle = data.qpos[self._door_qid]
    threshold = 0.5 * jp.pi - 0.01 # tolerance
    return angle >= threshold

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
        noisy_joint_angles,  # Hand joints
        info["last_act"],  # Previous actions
    ])
    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    # Get palm and handle positions
    palm_pos = data.site_xpos[self._palm_site_id]
    handle_pos = data.site_xpos[self._handle_site_id]
    palm_to_handle = palm_pos - handle_pos
    
    # Get door and latch angles
    door_angle = data.qpos[self._door_qid:self._door_qid+1]
    latch_angle = data.qpos[self._latch_qid:self._latch_qid+1]
    
    # Door open indicator
    door_open = jp.where(door_angle > 1.0, 1.0, -1.0)

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        data.actuator_force,
        palm_pos,
        handle_pos,
        palm_to_handle,
        door_angle,
        latch_angle,
        door_open,
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
    # Get palm and handle positions
    palm_pos = data.site_xpos[self._palm_site_id]
    handle_pos = data.site_xpos[self._handle_site_id]
    palm_to_handle_dist = jp.linalg.norm(palm_pos - handle_pos)
    
    # Door angle terms
    current_angle = data.qpos[self._door_qid]
    last_angle = jp.squeeze(info["last_door_angle"])  # shape () from (1,)
    delta_angle = current_angle - last_angle

    # Success proximity (same threshold as termination)
    threshold = 0.5 * jp.pi - 0.01
    door_open_event = jp.where(current_angle >= threshold, 1.0, 0.0)
    
    # Get velocities for penalty
    qvel = data.qvel
    
    return {
        "get_to_handle": -palm_to_handle_dist,  # Closer to current handle is better
        "velocity_penalty": jp.sum(jp.square(qvel)),  # Penalty for high velocities
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "door_angle": delta_angle,  # Positive if opening further this step
        "door_open": door_open_event,  # Big bonus when near 90 deg
    }

  def _get_door_bonus(self, door_angle: jax.Array) -> jax.Array:
    """Get bonus reward for door opening milestones."""
    bonus = jp.zeros(())
    bonus = jp.where(door_angle > 0.2, bonus + 2.0, bonus)
    bonus = jp.where(door_angle > 1.0, bonus + 8.0, bonus)
    bonus = jp.where(door_angle > 1.35, bonus + 10.0, bonus)
    return bonus

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))


def domain_randomize(model: mjx.Model, rng: jax.Array):
  mj_model = DoorOpen().mj_model
  hand_qids = mjx_env.get_qpos_ids(mj_model, ["H_Tx", "H_Rx", "H_Ry", "H_Rz"] + consts.JOINT_NAMES)
  hand_body_names = [
      "palm",
      "if_bs",
      "if_px",
      "if_md",
      "if_ds",
      "mf_bs",
      "mf_px",
      "mf_md",
      "mf_ds",
      "rf_bs",
      "rf_px",
      "rf_md",
      "rf_ds",
      "th_mp",
      "th_bs",
      "th_px",
      "th_ds",
  ]
  hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
  fingertip_geoms = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
  fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

  @jax.vmap
  def rand(rng):
    # Fingertip friction: =U(0.5, 1.0).
    rng, key = jax.random.split(rng)
    fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
    geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
        fingertip_friction
    )

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[hand_qids].set(
        qpos0[hand_qids]
        + jax.random.uniform(key, shape=(len(hand_qids),), minval=-0.05, maxval=0.05)
    )

    # Scale static friction: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
        key, shape=(len(hand_qids),), minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[hand_qids] * jax.random.uniform(
        key, shape=(len(hand_qids),), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[hand_qids].set(armature)

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[hand_body_ids].set(
        model.body_mass[hand_body_ids] * dmass
    )

    # Joint stiffness: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        key, (model.nu,), minval=0.8, maxval=1.2
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    # Joint damping: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    kd = model.dof_damping[hand_qids] * jax.random.uniform(
        key, (len(hand_qids),), minval=0.8, maxval=1.2
    )
    dof_damping = model.dof_damping.at[hand_qids].set(kd)

    return (
        geom_friction,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    )

  (
      geom_friction,
      body_mass,
      qpos0,
      dof_frictionloss,
      dof_armature,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  ) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_mass": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  return model, in_axes 