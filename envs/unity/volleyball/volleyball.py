OBS_SIZE = 49
LAYER_SIZE = 256
VISUALIZE = False
REW_TYPE = 'Lin'
DISCRETE_BINS = -1
NORMALIZE = True
NORM_TYPE = 'layer'

def create_ppo_agent(train_on=True):
    from ppo.ppo_params import PPOParams
    from ppo.ppo_agent import PPOAgent
    worker_ct = 1 if VISUALIZE else 40
    DPS = 60
    REW_HALF_LIFE = -1
    GAE_HALF_LIFE = 2
    agent_params = PPOParams(
        state_shape=[OBS_SIZE],
        act_sizes=[3, 3, 2, 3, 3, 2],
        save_path="output/Volleyball",
        use_act_mask=True,
        body_creator=get_body,
        val_head_creator=get_val_head,
        pol_head_creator=get_pol_head,
        pol_branch_creator=create_branch,
        discount=1 if REW_HALF_LIFE < 0 else 0.5**(1/(DPS*REW_HALF_LIFE)),
        gae_param=1 if GAE_HALF_LIFE < 0 else 0.5**(1/(DPS*GAE_HALF_LIFE)),
        entropy_coef=1e-2,
        buffer_size=2**20 if VISUALIZE else 2**13,
        batch_size=4096,
        learn_rate=5e-5,
        epoch_count=4,
        normalize_obs=False,
        discrete_bins=DISCRETE_BINS
    )
    return PPOAgent(agent_params, train_on=train_on, worker_count=worker_ct)

def create_bqn_agent(train_on=True):
    from bqn.bqn_params import BQNParams
    from bqn.bqn_agent import BQNAgent
    worker_ct = 1 if VISUALIZE else 40
    DPS = 15
    REW_HALF_LIFE = 10
    RANDOM_PERIOD = 5
    agent_params = BQNParams(
        state_shape=[OBS_SIZE],
        act_sizes=[3, 3, 2, 3, 3, 2],
        save_path="output/Volleyball",
        use_act_mask=True,
        body_creator=get_bqn_body,
        branch_creator=create_branch,
        discount=1 if REW_HALF_LIFE < 0 else 0.5**(1/(DPS*REW_HALF_LIFE)),
        buffer_size=2**20 if VISUALIZE else 2**11,
        learn_rate=1e-3,
        batch_per_train=2,
        train_per_step=1,
        sample_per_worker=1,
        target_update_period=2**10,
        random_act_prob=1.0/(RANDOM_PERIOD*DPS),
        q2pol_mul=5e1,
        val_momentum=1e-4,
    )
    return BQNAgent(agent_params, train_on=train_on, worker_count=worker_ct)

def get_bqn_body(state): return get_pol_head(get_body(state))

def get_body(state):
    from brain_tools.entity_rel import get_stacked_relations, get_relations
    import numpy as np
    # all_obs, ent_sizes, groups, types, layer_size
    player_size = 9
    ball_size = 12
    #player_player_size = player_size
    #player_ball_size = 7
    ent_sizes = []
    e_id = 0
    players = np.empty(4, dtype=np.int32)
    #player_player = np.empty((2, 3), dtype=np.int32)
    #player_ball = np.empty(2, dtype=np.int32)
    for p in range(4):
        ent_sizes.append(player_size)
        players[p] = e_id
        e_id += 1
    ent_sizes.append(ball_size)
    ball = e_id
    e_id += 1
    """
    for my_p in range(2):
        for ot_p in range(3):
            ent_sizes.append(player_player_size)
            player_player[my_p, ot_p] = e_id
            e_id += 1
        ent_sizes.append(player_ball_size)
        player_ball[my_p] = e_id
        e_id += 1
    """
    ent_sizes = np.array(ent_sizes, dtype=np.int32)
    print("ent_sizes: {}".format(ent_sizes))
    #print("players: {}\nball: {}\nplayer-player: {}\nplayer-ball: {}".format(players, ball, player_player, player_ball))
    print("players: {}\nball: {}".format(players, ball))
    groups = []
    types = []
    for my_p in range(2):
        for ot_p in range(3):
            if ot_p == 0:
                t = 0
                ot_id = 1 - my_p
            else:
                t = 1
                ot_id = ot_p + 1
            #g = [players[my_p], players[ot_id], player_player[my_p, ot_p]]
            g = [players[my_p], players[ot_id]]
            groups.append(g)
            types.append(t)
        #groups.append([players[my_p], ball, player_ball[my_p]])
        groups.append([players[my_p], ball])
        types.append(2)
    groups = np.array(groups)
    types = np.array(types)
    print("groups: {}\ntypes: {}".format(groups, types))
    body = get_relations(state, ent_sizes, groups, types, layer_size=LAYER_SIZE, normalize=NORMALIZE, norm_type=NORM_TYPE)
    #print("body: {}".format(body))
    return body

def norm_layer():
    if NORM_TYPE == 'layer':
        from brain_tools.layer_norm import LayerNorm
        return LayerNorm(scale=False)
    elif NORM_TYPE == 'batch':
        return 'batch_norm'
    else:
        print("Volleyball norm_layer(): Invalid norm type.")
        exit()

def get_stacked_val_head(body):
    from tensorflow.keras.layers import Dense, Activation, Flatten, Concatenate, MaxPool1D
    from brain_tools.entity_rel import apply_layers

    rels = body[0]
    global_obs = body[1]
    x = Concatenate()([
        Flatten()(MaxPool1D(2)(rels[0])),
        Flatten()(MaxPool1D(4)(rels[1])),
        Flatten()(MaxPool1D(2)(rels[2])),
    ])

    x = Activation('relu')(x)
    x = Concatenate()([x, global_obs])
    x = apply_layers(x, [
        Dense(LAYER_SIZE) if NORMALIZE else None,
        norm_layer() if NORMALIZE else None,
        Activation('relu') if NORMALIZE else None,
        Dense(LAYER_SIZE, activation='relu') if not NORMALIZE else None,
        Dense(LAYER_SIZE, activation='relu'),
        Dense(1, name='value')
    ])
    return x

def get_stacked_pol_head(body):
    from tensorflow.keras.layers import Concatenate, Dense, Activation, Flatten, MaxPool1D
    from brain_tools.entity_rel import apply_layers
    import tensorflow.keras.backend as K

    rels = body[0]
    global_obs = body[1]
    output = [None, None, [
               Dense(3),
               Dense(3),
               Dense(2)
    ]]

    stacked = []
    for p in range(2):
        if p == 0:
            x = Concatenate()([
                rels[0][:, 0, :],
                Flatten()(MaxPool1D(2)(rels[1][:, 0:2, :])),
                rels[2][:, 0, :]
            ])
        else:
            x = Concatenate()([
                rels[0][:, 1, :],
                Flatten()(MaxPool1D(2)(rels[1][:, 2:, :])),
                rels[2][:, 1, :]
            ])
        x = Activation('relu')(x)
        x = Concatenate()([x, global_obs])
        stacked.append(x)

    x = K.stack(stacked, axis=1)
    layers = [
        Dense(LAYER_SIZE) if NORMALIZE else None,
        norm_layer() if NORMALIZE else None,
        Activation('relu') if NORMALIZE else None,
        Dense(LAYER_SIZE, activation='relu') if not NORMALIZE else None,
        Dense(LAYER_SIZE, activation='relu')
    ]
    x = apply_layers(x, layers)
    for p in range(2):
        output[p] = x[:, p, :]

    return output

def get_val_head(body):
    from tensorflow.keras.layers import Concatenate, Dense, Activation
    from brain_tools.entity_rel import apply_layers, pool_and_concat

    x = pool_and_concat(body[0], [[0,4], [1,2,5,6], [3,7]])
    x = Activation('relu')(x)
    x = Concatenate()([x, body[1]])
    x = apply_layers(x, [
        Dense(LAYER_SIZE) if NORMALIZE else None,
        norm_layer() if NORMALIZE else None,
        Activation('relu') if NORMALIZE else None,
        Dense(LAYER_SIZE, activation='relu') if not NORMALIZE else None,
        Dense(LAYER_SIZE, activation='relu'),
        Dense(1, name='value')
    ])
    return x

def get_pol_head(body):
    from tensorflow.keras.layers import Concatenate, Dense, Activation
    from brain_tools.entity_rel import apply_layers, pool_and_concat

    rels = body[0]
    rel_count = len(rels)
    print("get_pol_head: rel_count: {}".format(rel_count))
    global_obs = body[1]
    output = [None, None, [
               Dense(3),
               Dense(3),
               Dense(2)
    ]]

    layers = [
        Dense(LAYER_SIZE) if NORMALIZE else None,
        norm_layer() if NORMALIZE else None,
        Activation('relu') if NORMALIZE else None,
        Dense(LAYER_SIZE, activation='relu') if not NORMALIZE else None,
        Dense(LAYER_SIZE, activation='relu')
    ]

    e = 0
    for p in range(2):
        s = e
        e += rel_count // 2
        x = rels[s:e]

        x = pool_and_concat(x, [0, [1, 2], 3])
        x = Activation('relu')(x)

        x = Concatenate()([x, global_obs])
        x = apply_layers(x, layers)
        output[p] = x

    return output

def create_branch(x, branch):
    b = branch
    if branch < 3:
        p = x[0]
    else:
        p = x[1]
        b = branch - 3
    return x[2][b](p)
