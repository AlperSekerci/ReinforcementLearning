from ppo.ppo_brain import PPOBrain
from ppo.tools import update_train_data, get_acts, rng
import numpy as np
import os

#region Training Meta
VAL_SCALE = 0
VAL_SHIFT = 1
META_CT = 2
META_FILE_NAME = 'meta.npy'
#endregion
class PPOAgent:
    def __init__(self, params, train_on=True, worker_count=1):
        self.params = params
        self.worker_ct = worker_count
        self.total_buf_size = worker_count * params.buffer_size
        self.train_on = train_on

        if train_on:
            self.buffer_state = np.empty([worker_count, params.buffer_size] + params.state_shape, dtype=np.float32)

            self.buffer_acts = [None] * params.branch_count
            for b in range(params.branch_count):
                self.buffer_acts[b] = np.empty((worker_count, params.buffer_size, params.act_sizes[b]), dtype=np.float32)

            if params.use_act_mask: self.buffer_mask = np.empty((worker_count, params.buffer_size, params.total_act_size), dtype=np.float32)
            self.buffer_reward = np.empty((worker_count, params.buffer_size), dtype=np.float32)
            self.buffer_done = np.empty((worker_count, params.buffer_size), dtype=np.bool)
            self.buffer_advantage = np.empty((worker_count, params.buffer_size), dtype=np.float32)
            self.buffer_return = np.empty((worker_count, params.buffer_size), dtype=np.float32)
            self.buffer_act_probs = np.empty((worker_count, params.buffer_size, params.branch_count), dtype=np.float32)
            self.buffer_pred_val = np.empty((worker_count, params.buffer_size), dtype=np.float32)

            self.buffer_pointer = 0
            self.buffer_filled = False
            self.rand_seq = np.arange(self.total_buf_size, dtype=np.int32)
            self.meta = np.zeros(META_CT, dtype=np.float32)

        self.brain = PPOBrain(params)

    def decide(self, obs, act_masks=None):
        if self.params.use_act_mask:
            model_in = [obs, act_masks]
        else: model_in = obs
        output = self.brain.fast_model(model_in)
        batch_size = obs.shape[0]

        values = np.array(output[0], dtype=np.float32)
        pols = output[1:]

        acts = np.empty((batch_size, self.params.branch_count), dtype=np.uint32)
        probs = np.empty((batch_size, self.params.branch_count), dtype=np.float32)
        get_acts(pols, acts, probs)
        return acts, probs, values

    def gain_exp(self, state, acts, act_probs, reward, done, pred_val, act_masks=None):
        if self.buffer_filled: return
        pointer = self.buffer_pointer
        self.buffer_state[:, pointer] = state

        for w in range(self.worker_ct):
            for b in range(self.params.branch_count):
                one_hot_act = np.zeros(self.params.act_sizes[b], dtype=np.float32)
                one_hot_act[acts[w][b]] = 1
                self.buffer_acts[b][w, pointer] = one_hot_act
                self.buffer_act_probs[w, pointer, b] = act_probs[w, b] + 1e-10

        if act_masks is not None:
            self.buffer_mask[:, pointer] = act_masks

        self.buffer_reward[:, pointer] = reward
        self.buffer_done[:, pointer] = done
        self.buffer_pred_val[:, pointer] = pred_val
        self.buffer_pointer += 1
        self.buffer_filled = self.buffer_pointer >= self.params.buffer_size

    def update_data(self):
        self.pre_data_update()
        update_train_data(self.buffer_reward, self.buffer_done, self.buffer_pred_val, self.buffer_advantage,
                          self.buffer_return, self.params.discount, self.params.gae_mul)
        self.post_data_update()

    def amplify_positive_rewards(self, mul=2):
        pos_mask = self.buffer_reward > 0
        self.buffer_reward[pos_mask] *= mul
        print("positive rewards are multiplied by {}".format(mul))

    def normalize_rewards_abs(self):
        avg_rew_mag = np.mean(np.abs(self.buffer_reward))
        self.buffer_reward /= avg_rew_mag
        np.clip(self.buffer_reward, -self.params.reward_clip, self.params.reward_clip, out=self.buffer_reward)
        print("average reward magnitude: {}\nreward clip: {}".format(avg_rew_mag, self.params.reward_clip))

    def pre_data_update(self):
        self.buffer_pred_val *= self.meta[VAL_SCALE]
        self.buffer_pred_val += self.meta[VAL_SHIFT]

    def post_data_update(self):
        adv_mean, adv_std = self.standardize_arr(self.buffer_advantage, 'advantage')
        ret_mean, ret_std = self.standardize_arr(self.buffer_return, 'return')
        self.meta[VAL_SCALE] = ret_std + self.params.epsilon
        self.meta[VAL_SHIFT] = ret_mean

    def standardize_arr(self, arr, name, clip=-1):
        mean = np.mean(arr)
        std = np.std(arr)
        arr -= mean
        arr /= std + self.params.epsilon
        if clip > 0:
            np.clip(arr, -clip, clip, out=arr)
            print("{} is clipped between -{} and {}".format(name, clip, clip))
        print("{} mean: {:.4f}\n{} std: {:.4f}".format(name, mean, name, std))
        return mean, std

    @staticmethod
    def normalize_nonzero_magnitude(arr, name, clip=-1.0):
        nonzero_mask = np.abs(arr) > 1e-10
        nonzero_arr = arr[nonzero_mask]
        print("non-zero {} count: {}".format(name, len(nonzero_arr)))
        mean = np.mean(np.abs(nonzero_arr))
        arr /= mean + 1e-10
        if clip > 0:
            np.clip(arr, -clip, clip, out=arr)
            print("{} is clipped between -{} and {} (max. magnitude: {})".format(name, clip, clip, clip * mean))
        print("{} absolute mean: {:.4f}".format(name, mean))

    def standardize_rewards(self):
        self.normalize_nonzero_magnitude(self.buffer_reward, 'reward', clip=1.5)

    def standardize_adv_mag(self):
        avg_mag = np.abs(self.buffer_advantage).mean()
        self.buffer_advantage /= avg_mag + 1e-10
        print("avg. advantage magnitude: {}".format(avg_mag))

    def standardize_returns(self):
        self.standardize_arr(self.buffer_return, 'return', clip=5)

    def is_ready_to_train(self):
        return self.buffer_filled

    def reset_buffer(self):
        self.buffer_pointer = 0
        self.buffer_filled = False

    def get_train_feed(self):
        rng.shuffle(self.rand_seq)
        state = self.buffer_state.reshape([self.total_buf_size] + self.params.state_shape)[self.rand_seq]
        adv = self.buffer_advantage.reshape(self.total_buf_size)[self.rand_seq]
        act_probs = self.buffer_act_probs.reshape((self.total_buf_size, self.params.branch_count))[self.rand_seq]
        ret = self.buffer_return.reshape(self.total_buf_size)[self.rand_seq]
        if self.params.use_act_mask:
            mask = self.buffer_mask.reshape((self.total_buf_size, self.params.total_act_size))[self.rand_seq]
        else: mask = None
        acts = []
        for b in range(self.params.branch_count):
            acts.append(self.buffer_acts[b].reshape(self.total_buf_size, self.params.act_sizes[b])[self.rand_seq])

        # PPOBrain train signature: state, act, old_act_probs, returns, advantage, mask=None
        return state, acts, act_probs, ret, adv, mask

    def train(self):
        if not self.is_ready_to_train(): return False
        self.update_data()
        feed = self.get_train_feed()
        self.brain.train(*feed, batch_size=self.params.batch_size, epoch_count=self.params.epoch_count)
        self.reset_buffer()
        return True

    def get_save_path(self, name, ext=""):
        save_folder = "./" + self.params.save_path + "/" + name
        return save_folder, save_folder + "/" + name + ext

    def save(self, name):
        save_folder, file_path = self.get_save_path(name)
        try: os.makedirs(save_folder)
        except: pass
        self.brain.model.save_weights(file_path)
        np.save("{}/{}".format(save_folder, META_FILE_NAME), self.meta)

    def restore(self, name, log=True):
        save_folder, file_path = self.get_save_path(name)
        if len(self.params.save_path) > 0 and os.path.isfile(file_path + ".index"):
            self.brain.model.load_weights(file_path)
            self.meta = np.load("{}/{}".format(save_folder, META_FILE_NAME))
            if log: print("restored {}".format(file_path))
        else:
            if log: print("PPOAgent: There is no weights file with name {} in folder {}.".format(name, save_folder))

    def export(self, name):
        save_folder, file_path = self.get_save_path(name)
        #self.brain.model.save(file_path, save_format='h5')

        # https://github.com/leimao/Frozen_Graph_TensorFlow/blob/master/TensorFlow_v2/train.py
        from tensorflow import function, TensorSpec
        from tensorflow.python.framework import graph_io as io
        # Convert Keras model to ConcreteFunction
        model = self.brain.model

        fn_inputs = []
        #print("model inputs:")
        for i in model.inputs:
            #print(i)
            spec = TensorSpec(i.shape, i.dtype, name=i.name)
            fn_inputs.append(spec)

        full_model = function(lambda x: model(x))
        full_model = full_model.get_concrete_function(fn_inputs)

        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        layers = [op.name for op in frozen_func.graph.get_operations()]
        #print("-" * 50)
        #print("Frozen model layers: ")
        #for layer in layers:
        #    print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

        # Save frozen graph from frozen ConcreteFunction to hard drive
        io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=save_folder,
                          name=name+".pb",
                          as_text=False)

        print("\n^^ EXPORTED {} ^^".format(name))
