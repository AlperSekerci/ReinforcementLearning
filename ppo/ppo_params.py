import numpy as np


class PPOParams:
    def __init__(self, state_shape, act_sizes,
                 hidden_layer_count=0,
                 hidden_neuron_count=0,
                 activation='relu',
                 val_activation=None,
                 learn_rate=1e-3,
                 policy_epsilon=0.2,
                 buffer_size=16384,
                 batch_size=256,
                 epoch_count=3,
                 discount=0.99,
                 gae_param=0.95,
                 entropy_coef=1e-2,
                 value_loss_coef=1,
                 save_path=".",
                 body_creator=None,
                 pol_head_creator=None,
                 pol_branch_creator=None,
                 val_head_creator=None,
                 use_act_mask=False,
                 normalize_obs=False,
                 discrete_bins=-1,
                 epsilon=1e-3,
                 ):

        self.state_shape = state_shape
        self.act_sizes = act_sizes
        self.branch_count = len(act_sizes)
        self.total_act_size = int(np.sum(act_sizes))
        self.hidden_layer_count = hidden_layer_count
        self.hidden_neuron_count = hidden_neuron_count
        self.activation = activation
        self.val_activation = val_activation
        self.learn_rate = learn_rate
        self.policy_epsilon = policy_epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.discount = discount
        self.gae_param = gae_param
        self.gae_mul = self.discount * self.gae_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.save_path = save_path
        self.body_creator = body_creator
        self.pol_head_creator = pol_head_creator
        self.pol_branch_creator = pol_branch_creator
        self.val_head_creator = val_head_creator
        self.use_act_mask = use_act_mask
        self.normalize_obs = normalize_obs
        self.discretize_obs = discrete_bins >= 2
        self.discrete_bins = discrete_bins
        self.epsilon = epsilon
