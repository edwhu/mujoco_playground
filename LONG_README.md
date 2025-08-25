

https://github.com/jax-ml/jax/issues/29843

```
unset LD_LIBRARY_PATH
```

Verify GPU

python -c "import jax; print(jax.default_backend())"


```
 CUDA_VISIBLE_DEVICES=0 python -u learning/train_jax_ppo.py --env_name=LeapCubeRotateZAxisTouch --domain_randomization --use_wandb --num_timesteps 1_000_000_000 --suffix touch 

 CUDA_VISIBLE_DEVICES=1 python -u learning/train_jax_ppo.py --env_name=LeapCubeRotateZAxis --domain_randomization --use_wandb --suffix notouch 
```


With randomization
```
CUDA_VISIBLE_DEVICES=0 python -u render_checkpoint.py  --checkpoint_dir /home/vlongle/code/mujoco_playground/logs/LeapCubeRotateZAxisTouch-20250820-200317-touch/checkpoints/ --env_name LeapCubeRotateZAxisTouch --enable_domain_randomization True --num_episodes 5
```

Without randomization
```
CUDA_VISIBLE_DEVICES=0 python -u render_checkpoint.py  --checkpoint_dir /home/vlongle/code/mujoco_playground/logs/LeapCubeRotateZAxisTouch-20250820-200317-touch/checkpoints/ --env_name LeapCubeRotateZAxisTouch --num_episodes 5
```


## Observation encoder Dev

DEBUG _get_obs:
  noisy_joint_angles.shape: (16,)
  touch.shape: (20,)
  info['last_act'].shape: (16,)
  state.shape after concatenate: (52,)
  obs_history.shape: (52,)
DEBUG privileged_state components:
  state.shape: (52,)
  joint_angles.shape: (16,)
  data.qvel[self._hand_dqids].shape: (16,)
  joint_torques.shape: (16,)
  fingertip_positions.shape: (12,)
  cube_pos_error.shape: (3,)
  cube_quat.shape: (4,)
  cube_angvel.shape: (3,)
  cube_linvel.shape: (3,)
  privileged_state.shape after concatenate: (125,)

============================================================
PPO NETWORK ARCHITECTURE
============================================================

PPO Configuration:
  Network factory: functools.partial(<function make_ppo_networks at 0x7f9964359a80>, policy_hidden_layer_sizes=(512, 256, 128), policy_obs_key='state', value_hidden_layer_sizes=(512, 256, 128), value_obs_key='privileged_state')
  Policy hidden layers: (512, 256, 128)
  Value hidden layers: (512, 256, 128)
  Policy observation key: state
  Value observation key: privileged_state
I0821 15:47:04.814189 140304427295104 io.py:153] Using JAX default device: cuda:0.

Network Input/Output:
  Observation keys: ['privileged_state', 'state']
  Action space size: 16
  privileged_state shape: (1, 125)
  state shape: (1, 56)
  Could not inspect detailed network structure: cannot unpack non-iterable PPONetworks object
Traceback (most recent call last):
  File "/home/vlongle/code/mujoco_playground/inspect_checkpoint.py", line 199, in print_ppo_architecture
    policy_network, value_network = network_factory(obs_size, env.action_size, preprocess_observations_fn=lambda x: x)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: cannot unpack non-iterable PPONetworks object

============================================================
INSPECTION COMPLETE
============================================================