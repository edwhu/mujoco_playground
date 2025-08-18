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

def _get_frame_body_id() -> int:
  """Get frame body ID - hardcoded for now since we can't access mj_model from domain_randomize."""
  # Frame body ID is 21 (found by testing)
  # This is hardcoded because domain_randomize only receives mjx_model, not mj_model
  return 19


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
              translation_velocity_penalty=1e-2,  # Additional penalty for translation joints
              action_rate=0.0,
              door_angle=1.0,
              door_open=100.0,
          ),
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
        xml_path="mujoco_playground/_src/manipulation/leap_hand/xmls/door_scene.xml",
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    # Get hand joint IDs (including base motion joints) - only hand joints are controllable
    hand_joint_names = ["H_Tx", "H_Ty", "H_Rx", "H_Ry", "H_Rz"] + consts.JOINT_NAMES
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, hand_joint_names)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, hand_joint_names)
    
    # Get door and latch joint IDs (these are not controllable)
    self._door_qid = mjx_env.get_qpos_ids(self.mj_model, ["door_hinge"])[0]
    self._latch_qid = mjx_env.get_qpos_ids(self.mj_model, ["latch"])[0]
    
    # Get site IDs for palm and handle
    self._palm_site_id = self._mj_model.site("palm_site").id
    self._handle_site_id = self._mj_model.site("S_handle").id
    
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
    # Explicitly set door and latch closed at start
    qpos = qpos.at[self._door_qid].set(0.0)
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

    # State size is 21 hand joints + 21 previous actions = 42
    state_size = len(self._hand_qids) + self.mjx_model.nu
    obs_history = jp.zeros(self._config.history_len * state_size)
    obs = self._get_obs(data, info, obs_history)
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Clip actions to actuator control ranges before scaling
    motor_targets = self._default_pose + action
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    
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
        noisy_joint_angles,  # Hand joints (21 values: 5 base motion + 16 finger joints)
        info["last_act"],  # Previous actions (21 values)
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
    
    # Fingertip positions
    fingertip_positions = self.get_fingertip_positions(data)

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        fingertip_positions,
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
        "velocity_penalty": -jp.sum(jp.square(qvel)),  # Penalty for high velocities
        "translation_velocity_penalty": -jp.sum(jp.square(qvel[0:2])), # Penalty for high translation velocities
        "action_rate": -self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "door_angle": delta_angle,  # Positive if opening further this step
        "door_open": door_open_event,  # Big bonus when near 90 deg
    }

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))



def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Randomizes door position by modifying frame body position."""

  # Get frame body ID (hardcoded for now)
  frame_body_id = _get_frame_body_id()

  @jax.vmap
  def randomize_door_pos(rng):
    # Sample continuous frame offsets within joint ranges
    rng, kx, ky, kz = jax.random.split(rng, 4)
    tx = jax.random.uniform(kx, (), minval=-0.5, maxval=0.5)
    ty = jax.random.uniform(ky, (), minval=-0.2, maxval=0.2)
    # tz = jax.random.uniform(kz, (), minval=-0.00, maxval=0.00)
    
    # Get the original frame position and add the offset
    original_pos = model.body_pos[frame_body_id]
    offset = jp.array([tx, ty, 0.0])
    new_pos = original_pos + offset

    # Create new body_pos with randomized frame position
    new_body_pos = model.body_pos.at[frame_body_id].set(new_pos)
    return new_body_pos

  body_pos = randomize_door_pos(rng)

  in_axes = jax.tree_util.tree_map(lambda _: None, model)
  in_axes = in_axes.tree_replace({
      "body_pos": 0,
  })

  model = model.tree_replace({
      "body_pos": body_pos,
  })

  return model, in_axes
