

https://github.com/jax-ml/jax/issues/29843

```
unset LD_LIBRARY_PATH
```

Verify GPU

python -c "import jax; print(jax.default_backend())"


```
 CUDA_VISIBLE_DEVICES=0 python -u learning/train_jax_ppo.py --env_name=LeapDoorOpenTouchSimple --domain_randomization --use_wandb --num_timesteps 1000000000 --suffix touch

 CUDA_VISIBLE_DEVICES=1 python -u learning/train_jax_ppo.py --env_name=LeapDoorOpen --domain_randomization --use_wandb --num_timesteps 1000000000 --suffix notouch
```