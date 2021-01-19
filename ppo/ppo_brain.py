from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow import math, function, GradientTape, one_hot
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from time import time as get_time
import numpy as np
from brain_tools.obs_norm import ObsNormalizer
from tensorflow.keras.layers.experimental.preprocessing import Discretization
from scipy.stats import norm

np.set_printoptions(suppress=True, formatter={'float_kind': '{:.3f}'.format})


class PPOBrain:
    def __init__(self, params):
        self.params = params
        #get_logger().setLevel('ERROR')

        #region Input
        self.org_state = Input(shape=params.state_shape, name="state", dtype='float32')
        if params.normalize_obs:
            self.obs_normalizer = ObsNormalizer()
            self.state = self.obs_normalizer(self.org_state)
        else: self.state = self.org_state
        if params.discretize_obs: self.state = self.discretize_obs(self.state)

        if params.use_act_mask:
            self.act_masks = [None] * params.branch_count
            self.combined_mask = Input(shape=params.total_act_size, name="combined_mask", dtype='float32')
            act_index = 0
            for i in range(params.branch_count):
                act_end = act_index + params.act_sizes[i]
                self.act_masks[i] = self.combined_mask[:, act_index:act_end]
                act_index = act_end
        #endregion

        #region Actor & Critic
        self.body = self.state if params.body_creator is None else params.body_creator(self.state)

        self.policy_head = self.create_head(self.body) if params.pol_head_creator is None else params.pol_head_creator(self.body)
        self.pols = [None] * params.branch_count
        for b in range(params.branch_count):
            self.pols[b] = self.create_policy(self.policy_head, branch=b)

        if self.params.val_head_creator is None:
            value_head = self.create_head(self.body)
            self.v = Dense(1, activation=params.val_activation, name="value")(value_head)
        else: self.v = params.val_head_creator(self.body)
        self.v = self.v[:, 0]
        # endregion

        #region Model & Training
        if params.use_act_mask: model_inputs = [self.org_state, self.combined_mask]
        else: model_inputs = self.org_state
        self.model = Model(inputs=model_inputs, outputs=[self.v] + self.pols)
        self.fast_model = function(self.model)
        self.optimizer = Adam(learning_rate=params.learn_rate)
        #endregion

    def discretize_obs(self, obs):
        bin_count = self.params.discrete_bins
        strategy = 'uniform'
        if strategy == 'quantile':
            bin_width = 1.0 / bin_count
            bins = (np.arange(bin_count - 1) + 1) * bin_width
            bins = norm.ppf(bins)
        elif strategy == 'uniform':
            z_95 = norm.ppf(0.95)
            z_05 = -z_95
            bin_width = (z_95 - z_05) / (bin_count - 2)
            bins = np.arange(bin_count - 1) * bin_width + z_05
            print("uniform bins (between 5% and 95%, bin count: {}): {}".format(bin_count, bins))
        else:
            print("PPOBrain: Invalid discretization strategy: {}".format(strategy))
            exit()
        discretizer = Discretization(bins=bins)
        x = discretizer(obs)
        x = one_hot(x, bin_count)
        x = Flatten()(x)
        return x

    @staticmethod
    @function
    def val_loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred))

    @function
    def pol_loss(self, branch, onehot_act, pol, old_act_prob, advantage):
        act_prob = K.sum(onehot_act * pol, axis=-1)
        ratio = act_prob / old_act_prob
        lower_bound = 1 - self.params.policy_epsilon
        upper_bound = 1 + self.params.policy_epsilon
        mid_point = 1 / self.params.act_sizes[branch]
        entropy_cost = (mid_point - act_prob) * self.params.entropy_coef
        edited_adv = advantage + entropy_cost
        no_clip = ratio * edited_adv
        clipped = K.clip(ratio, lower_bound, upper_bound) * edited_adv
        return -K.mean(K.minimum(no_clip, clipped))

    def create_head(self, body):
        hidden_layer = body
        for i in range(self.params.hidden_layer_count):
            hidden_layer = Dense(self.params.hidden_neuron_count, activation=self.params.activation)(hidden_layer)
        return hidden_layer

    def create_policy(self, input, branch):
        if self.params.pol_branch_creator is None:
            policy_logits = Dense(self.params.act_sizes[branch])(input)
        else: policy_logits = self.params.pol_branch_creator(input, branch)

        if self.params.use_act_mask: a = K.exp(policy_logits) * self.act_masks[branch]
        else: a = K.exp(policy_logits)
        sum = K.sum(a, axis=-1, keepdims=True)
        policy = math.divide(a, sum, name="policy_{}".format(branch))
        return policy

    def __str__(self):
        return str(self.model.summary())

    def plot(self, image_path):
        plot_model(self.model, image_path, show_shapes=True, rankdir='LR')

    def update_obs_norm(self, obs, log=True):
        mean = np.mean(obs, axis=0)
        std = np.std(obs, axis=0)
        shift = -mean
        scale = 1.0 / (std + self.params.epsilon)
        self.obs_normalizer.set_weights([shift, scale])
        if log:
            #print("obs mean: {}\nobs std:  {}".format(mean, std))
            print("PPOBrain: ObsNormalizer is updated.")

    def train(self, state, act, old_act_probs, returns, advantage, mask=None, epoch_count=1, batch_size=32):
        print("PPOBrain: Training is now starting.")
        buffer_size = state.shape[0]
        batch_per_epoch = buffer_size // batch_size
        print("buffer size: {}\nbatch size: {}\nbatch per epoch: {}\nlearning rate: {}".format(buffer_size, batch_size, batch_per_epoch, self.params.learn_rate))
        if self.params.normalize_obs: self.update_obs_norm(state)
        for epoch in range(epoch_count):
            print("Epoch {}/{}:".format(epoch + 1, epoch_count), end='')
            time = get_time()
            epoch_val_loss = 0
            epoch_pol_loss = [0] * self.params.branch_count
            e = 0
            for batch_idx in range(batch_per_epoch):
                s = e
                e += batch_size
                if self.params.use_act_mask:
                    model_in = [state[s:e], mask[s:e]]
                else: model_in = state[s:e]
                with GradientTape() as tape:
                    out = self.fast_model(model_in, training=True)
                    v = out[0]
                    pols = out[1:]

                    val_loss = self.val_loss(returns[s:e], v)
                    epoch_val_loss += val_loss
                    total_loss = val_loss * self.params.value_loss_coef

                    for b in range(self.params.branch_count):
                        pol_loss = self.pol_loss(b,
                            act[b][s:e],
                            pols[b],
                            old_act_probs[s:e][:, b],
                            advantage[s:e],
                        )
                        epoch_pol_loss[b] += pol_loss
                        total_loss += pol_loss
                grads = tape.gradient(total_loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            epoch_str = " losses: value: {:.5f}".format(epoch_val_loss)
            for b in range(self.params.branch_count):
                epoch_str += " b{}: {:.5f}".format(b, epoch_pol_loss[b])

            start_time = time
            time = get_time()
            delta_time = time - start_time
            epoch_str += ", {:.3f} seconds.".format(delta_time)
            print(epoch_str)
