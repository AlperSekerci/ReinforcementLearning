import numpy as np
import numba as nb

from numpy.random import default_rng
rng = default_rng()


@nb.njit(fastmath=True, parallel=True)
def update_train_data(buf_rew, buf_done, buf_val, buf_adv, buf_ret, discount, gae_mul):
    worker_ct = buf_rew.shape[0]
    buf_size = buf_rew.shape[1]
    last_idx = buf_size - 1

    for w in nb.prange(worker_ct):
        buf_ret[w, last_idx] = buf_val[w, last_idx]
        buf_adv[w, last_idx] = 0

        for a in range(last_idx - 1, -1, -1):
            ret = buf_rew[w, a]
            adv = -buf_val[w, a] + ret
            next_exp = a + 1
            if not buf_done[w, a]:
                future_avg_rew = discount * buf_val[w, next_exp]
                adv += future_avg_rew + gae_mul * buf_adv[w, next_exp]
                ret += discount * buf_ret[w, next_exp]
            buf_adv[w, a] = adv
            buf_ret[w, a] = ret

def get_acts(pols, acts, probs):
    worker_ct = acts.shape[0]
    branch_ct = acts.shape[1]
    for b in range(branch_ct):
        branch_pol = np.array(pols[b], dtype=np.float32)
        branch_acts = random_choice(branch_pol)
        acts[:, b] = branch_acts
        probs[:, b] = branch_pol[(np.arange(worker_ct), branch_acts)]

def random_choice(probs):
    return (probs.cumsum(1) > rng.uniform(size=probs.shape[0])[:, None]).argmax(1)

@nb.njit(fastmath=True, parallel=True)
def clip_excess_adv(buf_adv, act_probs, act_sizes, adv_cost):
    branch_ct = len(act_sizes)
    prob_midpoint = [1.0] * branch_ct
    for b in nb.prange(branch_ct):
        prob_midpoint[b] /= act_sizes[b]
    buf_size = len(buf_adv)
    up_count = 0
    down_count = 0
    for i in nb.prange(buf_size):
        adv = buf_adv[i]
        excessive_up = adv > 0
        excessive_down = adv < 0
        for b in range(branch_ct):
            prob = act_probs[i][b]
            if prob < prob_midpoint[b] and adv > 0:
                excessive_up = False
            elif prob > prob_midpoint[b] and adv < 0:
                excessive_down = False
        if excessive_up:
            adv -= adv_cost
            if adv < 0: adv = 0
            up_count += 1
        elif excessive_down:
            adv += adv_cost
            if adv > 0: adv = 0
            down_count += 1
        buf_adv[i] = adv
    return up_count, down_count

@nb.njit
def standardize(arr):
    if len(arr) == 0: return 0, 0
    mean = np.mean(arr)
    std = np.std(arr) + 1e-10
    arr[:] = (arr - mean) / std
    return mean, std
