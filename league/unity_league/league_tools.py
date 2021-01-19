def my_unity_worker(port, game_path, agent_params, acts, g_step, ready, reset_mode, obs, mask, rew, done, result):
    from envs.unity.my_unity.my_unity_env import MyUnityEnv
    import numpy as np
    import time

    env = MyUnityEnv(port, game_path, agent_params, reset_mode.value)
    step = 0

    acts = np.frombuffer(acts.get_obj(), dtype=np.ubyte).reshape((2, agent_params.branch_count))
    obs = np.frombuffer(obs.get_obj(), dtype=np.float32).reshape([2] + agent_params.state_shape)

    obs[:] = env.obs
    if agent_params.use_act_mask:
        mask = np.frombuffer(mask.get_obj(), dtype=np.float32).reshape((2, agent_params.total_act_size))
        mask[:] = env.mask

    ready.value = True
    while True:
        if step == g_step.value:
            time.sleep(0)
            continue
        step += 1

        env.step(acts, reset_mode.value)
        obs[:] = env.obs
        if agent_params.use_act_mask: mask[:] = env.mask
        rew.value = env.rew
        done.value = env.done
        result.value = env.result
        ready.value = True
