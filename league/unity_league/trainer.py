import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import time
import numpy as np
from league import League
from tools import print_prog_bar
from train_tools import get_rand_acts
from unity_league.league_tools import my_unity_worker
import cProfile, pstats, io

PPO = 0
BQN = 1


class Trainer:
    def __init__(self, base_port, game_path,
                 create_agent,
                 grind_step=64,
                 env_name="noname",
                 ignore_draw=False,
                 ranked_reset=0,
                 default_reset=0,
                 elo_interval=150,
                 king_count=4,
                 use_chal_as_op=False,
                 print_obs=None,
                 reward_mul=1,
                 visualize=False,
                 use_profiler=False,
                 profiler_name="profiler_result",
                 algo='ppo',
                 plot_brain=False,
                 ):

        self.base_port = base_port
        self.game_path = game_path
        self.create_agent = create_agent
        self.grind_step = grind_step
        self.env_name = env_name
        self.ignore_draw = ignore_draw
        self.ranked_reset = ranked_reset
        self.default_reset = default_reset
        self.print_obs = print_obs
        self.reward_mul = reward_mul
        self.visualize = visualize
        self.plot_brain = plot_brain

        self.use_profiler = use_profiler
        self.profiler = None
        self.train_loop_ct = 0
        self.profiler_used = False
        self.profiler_name = profiler_name

        algo = algo.lower()
        if algo == 'ppo': self.algo = PPO
        elif algo == 'bqn': self.algo = BQN
        else:
            print("Training algo. {} is invalid.".format(algo))
            exit()

        self.test_ep_rew = 0
        self.test_prev_val = 0

        self.league = None
        self.king_count = king_count + 1 if use_chal_as_op else king_count
        self.history_match_ct = 0
        self.self_match_ct = 0
        self.king_elo = 0
        self.elo_interval = elo_interval
        self.use_chal_as_op = use_chal_as_op

        self.agent_params = None
        self.use_mask = False
        self.valid_match = None
        self.scores = None
        self.req_scores = None

        self.workers = None
        self.worker_ct = 0
        self.g_step = None
        self.reset_mode = None
        self.ready = []
        self.acts = [[], []]
        self.obs = [[], []]
        self.mask = [[], []]
        self.rew = []
        self.done = []
        self.result = []
        self.rew_buf = None
        self.done_buf = None
        self.op_decided_act = None

        self.total_env_rew = 0.0
        self.total_ep_ct = 0
        self.total_act_prob = None # numpy array

    def init_league(self):
        self.league = League(self.create_agent, "output/{}".format(self.env_name))
        self.worker_ct = self.league.challenger.worker_ct
        self.agent_params = self.league.challenger.params
        self.use_mask = self.agent_params.use_act_mask

        if self.use_chal_as_op:
            self.history_match_ct = self.agent_params.buffer_size // 4
            self.self_match_ct = self.history_match_ct * 3
            self.history_match_ct = self.history_match_ct // (self.king_count - 1)
            self.king_elo = self.elo_interval * (self.king_count - 1)
        else:
            self.history_match_ct = self.agent_params.buffer_size // self.king_count
            self.king_elo = self.elo_interval * self.king_count

        self.total_act_prob = np.zeros(self.agent_params.branch_count)
        self.valid_match = np.zeros(self.worker_ct, dtype=np.bool)
        self.scores = np.zeros((self.king_count, 2), dtype=np.float32)
        self.req_scores = np.empty(self.king_count, dtype=np.float32)

        for i in range(self.king_count):
            op_elo = i * self.elo_interval
            self.req_scores[i] = self.league.calc_expected_result(self.king_elo, op_elo)

        # region Workers
        from multiprocessing import Process, Value, Array
        import ctypes

        self.workers = []
        self.g_step = Value(ctypes.c_uint64, 0)
        self.reset_mode = Value(ctypes.c_uint8, self.default_reset)
        act_buf_size = int(2 * self.agent_params.branch_count)
        obs_buf_size = int(2 * np.prod(self.agent_params.state_shape))
        mask_buf_size = int(2 * self.agent_params.total_act_size)
        self.rew_buf = np.empty(self.worker_ct, dtype=np.float32)
        self.done_buf = np.empty(self.worker_ct, dtype=np.bool)
        self.op_decided_act = np.empty((self.worker_ct, self.agent_params.branch_count), dtype=np.uint32)

        for w in range(self.worker_ct):
            ready = Value(ctypes.c_bool, False)
            acts = Array(ctypes.c_uint8, act_buf_size)
            obs = Array(ctypes.c_float, obs_buf_size)
            rew = Value(ctypes.c_float, 0)
            done = Value(ctypes.c_bool, False)
            result = Value(ctypes.c_int8, -2)

            np_acts = np.frombuffer(acts.get_obj(), dtype=np.ubyte).reshape((2, self.agent_params.branch_count))
            self.acts[0].append(np_acts[0])
            self.acts[1].append(np_acts[1])

            np_obs = np.frombuffer(obs.get_obj(), dtype=np.float32).reshape([2] + self.agent_params.state_shape)
            self.obs[0].append(np_obs[0])
            self.obs[1].append(np_obs[1])

            if self.use_mask:
                mask = Array(ctypes.c_float, mask_buf_size)
                np_mask = np.frombuffer(mask.get_obj(), dtype=np.float32).reshape((2, self.agent_params.total_act_size))
                self.mask[0].append(np_mask[0])
                self.mask[1].append(np_mask[1])
            else:
                mask = None

            self.ready.append(ready)
            self.rew.append(rew)
            self.done.append(done)
            self.result.append(result)

            # port, game_path, agent_params, acts, g_step, ready, reset_mode, obs, mask, rew, done, result
            worker = Process(target=my_unity_worker, args=(
                self.base_port + w,
                self.game_path,
                self.agent_params,
                acts,
                self.g_step,
                ready,
                self.reset_mode,
                obs,
                mask,
                rew,
                done,
                result
            ))

            self.workers.append(worker)
            worker.start()

        self.wait_workers()
        # endregion

    def wait_workers(self):
        while True:
            can_cont = True
            for w in range(self.worker_ct):
                if not self.ready[w].value:
                    can_cont = False
                    break
            if can_cont: break
            time.sleep(0)

    def __train_1v1(self, op_idx, king_id, match_count):
        my_team = 0
        op_team = 1 - my_team
        half_match_count = match_count // 2

        for step in range(match_count):
            self.reset_mode.value = self.ranked_reset if step >= half_match_count else self.default_reset

            decide_my_obs = np.copy(self.obs[my_team])
            if self.use_mask:
                decide_my_mask = np.copy(self.mask[my_team])
                decide_op_mask = np.array(self.mask[op_team])
            else:
                decide_my_mask = None
                decide_op_mask = None

            if op_idx == self.league.get_agent_count():
                op_agent = self.league.challenger
                agent_name = 'Self'
            else:
                op_agent = self.league.get_frozen_agent(op_idx)
                agent_name = 'Random' if op_idx <= 0 else "{} [{} elo]".format(op_idx, op_idx * self.elo_interval)

            if self.algo == PPO: my_decided_act, my_probs, my_values = self.league.challenger.decide(decide_my_obs, act_masks=decide_my_mask)
            elif self.algo == BQN: my_decided_act, my_probs = self.league.challenger.decide(decide_my_obs, act_masks=decide_my_mask)
            if op_agent is None:
                get_rand_acts(self.op_decided_act, self.agent_params.act_sizes, decide_op_mask)
            else:
                decide_op_obs = np.array(self.obs[op_team], dtype=np.float32)
                if self.algo == PPO: self.op_decided_act, op_probs, op_values = op_agent.decide(decide_op_obs, act_masks=decide_op_mask)
                elif self.algo == BQN: self.op_decided_act, op_probs = op_agent.decide(decide_op_obs, act_masks=decide_op_mask)

            for w in range(self.worker_ct):
                self.acts[my_team][w][:] = my_decided_act[w].astype(np.ubyte)
                self.acts[op_team][w][:] = self.op_decided_act[w].astype(np.ubyte)
                self.ready[w].value = False
            self.g_step.value += 1
            self.wait_workers()

            self.total_act_prob += np.sum(my_probs, axis=0)
            for w in range(self.worker_ct):
                self.rew_buf[w] = self.rew[w].value * self.reward_mul
                self.done_buf[w] = self.done[w].value

            if self.visualize:            
                self.test_ep_rew += self.rew_buf[0]
                if self.algo == PPO:
                    test_adv = self.rew_buf[0] - self.test_prev_val
                    if not self.done_buf[0]: test_adv += self.agent_params.discount * my_values[0]
                    self.test_prev_val = my_values[0]
                    print("episode rew: {:.3f} my_value: {:.3f} adv: {:.3f} act_prob_0: {:.2f} act_prob_1: {:.2f}".format(
                        self.test_ep_rew, my_values[0], test_adv, my_probs[0][0], my_probs[0][1]))
                elif self.algo == BQN:
                    print("episode rew: {:.3f}".format(self.test_ep_rew))
                if self.done_buf[0]: self.test_ep_rew = 0
                time.sleep((2/15))
                #self.print_obs(decide_my_obs[0], "my")
                #print("op act: {}".format(self.op_decided_act[0]))
                #print("op mask: {}".format(decide_op_mask))
                #print("my mask: {}\n".format(decide_my_mask))
                #print("\nobs: {}\nact: {}\nrew: {}\ndone: {}\n".format(decide_my_obs, my_decided_act, self.rew_buf, self.done_buf))

            if self.algo == PPO:
                self.league.challenger.gain_exp(decide_my_obs, my_decided_act, my_probs,
                                                np.copy(self.rew_buf), np.copy(self.done_buf), my_values, decide_my_mask)
            elif self.algo == BQN:
                self.league.challenger.gain_exp(decide_my_obs, my_decided_act, np.copy(self.rew_buf), np.copy(self.done_buf), decide_my_mask)
                self.league.challenger.train()

            for w in range(self.worker_ct):
                self.total_env_rew += self.rew_buf[w]
                if self.done_buf[w]:
                    self.total_ep_ct += 1
                    if self.valid_match[w]:
                        score = 0
                        draw = False
                        if self.result[w].value == my_team:
                            score = 1
                        elif self.result[w].value == -1:
                            score = 0.5
                            draw = True

                        ignore = self.ignore_draw and draw
                        if not ignore:
                            self.scores[king_id, 0] += score
                            self.scores[king_id, 1] += 1
                    self.valid_match[w] = self.reset_mode.value == self.ranked_reset
            if not self.visualize: print_prog_bar("Grinding {}".format(agent_name), step + 1, match_count, self.grind_step)

        # After steps
        self.valid_match.fill(False)

    def __get_king_avg_score(self, king_idx):
        score = self.scores[king_idx, 0]
        match_count = self.scores[king_idx, 1]
        if match_count > 0: score /= match_count
        return score

    def train(self):
        if self.use_profiler: self.setup_profiler()
        self.init_league()
        if self.plot_brain: self.league.challenger.brain.plot("output/brain.png")
        while True:
            using_profiler = self.use_profiler and not self.profiler_used and self.train_loop_ct == 1
            if using_profiler:
                self.profiler.enable()
                print("profiler started")
            time_start = time.time()

            agent_count = self.league.get_agent_count()
            if self.use_chal_as_op:
                valid_agent_ct = min(agent_count + 1, self.king_count)
                first_op_idx = agent_count
            else:
                valid_agent_ct = min(agent_count, self.king_count)
                first_op_idx = agent_count - 1
            lower_bound = first_op_idx - self.king_count
            weakest_king_idx = lower_bound + 1
            for op_idx in range(first_op_idx, lower_bound, -1):
                if self.use_chal_as_op and op_idx == agent_count: match_ct = self.self_match_ct
                else: match_ct = self.history_match_ct
                fixed_op_idx = max(op_idx, 0)
                king_idx = max(fixed_op_idx - weakest_king_idx, 0)
                self.__train_1v1(fixed_op_idx, king_idx, match_ct)

            if self.total_ep_ct == 0: avg_rew = 0
            else: avg_rew = self.total_env_rew / self.total_ep_ct
            avg_act_prob = np.around(self.total_act_prob / self.league.challenger.total_buf_size, 4)
            print("avg. reward: {:.4f}\ntotal episode count: {}\navg. action probability: {}".format(avg_rew, self.total_ep_ct, avg_act_prob))
            if self.algo == BQN: print("q_norm shift & scale: {}".format(self.league.challenger.brain.get_active_norm_stats()))
            self.total_env_rew = 0
            self.total_ep_ct = 0
            self.total_act_prob.fill(0)

            promoted = True
            index_offset = self.king_count - valid_agent_ct

            for i in range(valid_agent_ct):
                index = index_offset + i
                score = self.__get_king_avg_score(index)
                match_count = self.scores[index, 1]
                req_score = self.req_scores[index]
                agent_idx = max(first_op_idx - (valid_agent_ct - i) + 1, 0)
                agent_elo = agent_idx * self.elo_interval
                if self.use_chal_as_op and i == valid_agent_ct - 1:
                    passed = True
                    pass_text = "NOT IMPORTANT (PASSED by default)"
                    agent_name = 'Self'
                else:
                    passed = score >= req_score
                    pass_text = 'PASSED' if passed else 'FAILED'
                    if agent_idx <= 0: agent_name = 'Random'
                    else: agent_name = "{} [{} elo]".format(agent_idx, agent_elo)
                print("Score vs. {}: {:.4f} (in {} matches), Required: {:.4f}, {}".format(
                    agent_name, score, int(match_count), req_score, pass_text))
                if not passed: promoted = False

            if promoted: self.league.promote()
            if self.algo == PPO: self.league.challenger.train()
            self.league.challenger.save('challenger')
            time_now = time.time()
            dt = time_now - time_start
            print("One iteration of training lasted {} seconds.\n".format(dt))
            self.scores.fill(0)
            self.train_loop_ct += 1
            if using_profiler:
                self.profiler.disable()
                s = io.StringIO()
                sortby = 'cumulative'
                ps = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
                ps.dump_stats("{}.dmp".format(self.profiler_name))
                self.profiler_used = True
                print("profiler ended")

    def setup_profiler(self):
        if not self.use_profiler: return
        self.profiler = cProfile.Profile()
