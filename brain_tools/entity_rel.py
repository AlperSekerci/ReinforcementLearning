import numpy as np
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Maximum, Concatenate
from brain_tools.layer_norm import LayerNorm
import tensorflow.keras.backend as K

def get_stacked_relations(all_obs, ent_sizes, groups, types, layer_size=128, normalize=False, norm_type='layer'):
    group_ct = len(groups)
    if group_ct != len(types):
        print("get_relations: 'types' has not the same length as 'groups'")
        exit()
    entities = get_entities(all_obs, ent_sizes)
    groups = get_groups(entities, groups)
    type_ct = np.max(types) + 1

    rels = []
    sep_by_types = []
    for _ in range(type_ct): sep_by_types.append([])
    for group_idx in range(group_ct):
        group_type = types[group_idx]
        sep_by_types[group_type].append(groups[group_idx])

    for type_idx in range(type_ct):
        type_rels = sep_by_types[type_idx]
        rel_ct = len(type_rels)
        if rel_ct > 1:
            x = K.stack(type_rels, axis=1)
        elif rel_ct == 0:
            x = type_rels[0]
        else:
            print("get_stacked_relations(): type {} has no elements".format(type_idx))
            rels.append(None)
            continue
        x = apply_layers(x, get_rel_layers(layer_size, normalize, norm_type))
        rels.append(x)

    global_obs = entities[-1]
    return rels, global_obs

def get_relations(all_obs, ent_sizes, groups, types, layer_size=128, normalize=False, norm_type='layer'):
    group_ct = len(groups)
    if group_ct != len(types):
        print("get_relations: 'types' has not the same length as 'groups'")
        exit()
    entities = get_entities(all_obs, ent_sizes)
    groups = get_groups(entities, groups)
    type_ct = np.max(types) + 1
    type_layers = [None] * type_ct
    rels = []
    for group_idx in range(group_ct):
        group_type = types[group_idx]
        if type_layers[group_type] is None:
            type_layers[group_type] = get_rel_layers(size=layer_size, normalize=normalize, norm_type=norm_type)
        r = apply_layers(groups[group_idx], type_layers[group_type])
        rels.append(r)
    global_obs = entities[-1]
    return rels, global_obs

def apply_layers(x, layers):
    out = x
    for layer in layers:
        if layer is None: continue
        elif layer == 'batch_norm': out = BatchNormalization(scale=False)(out)
        else: out = layer(out)
    return out

def get_rel_layers(size, normalize=False, norm_type='layer'):
    layers = []
    if normalize:
        layers += [
        Dense(size),
        LayerNorm(scale=False) if norm_type == 'layer' else 'batch_norm',
        Activation('relu'),
        ]
    else:
        layers += [
            Dense(size, activation='relu')
        ]
    layers += [
        Dense(size, activation='relu'),
        Dense(size),
    ]
    return layers

def get_groups(entities, groups):
    from tensorflow.keras.layers import Concatenate
    data = []
    for group in groups:
        members = []
        for ent_idx in group:
            members.append(entities[ent_idx])
        data.append(Concatenate()(members))
    return data

def get_entities(all_obs, sizes):
    data = []
    e = 0
    for size in sizes:
        s = e
        e += size
        ent = all_obs[:, s:e]
        data.append(ent)
    if e < all_obs.shape[-1]:
        global_obs = all_obs[:, e:]
        data.append(global_obs)
    return data

def pool_and_concat(rels, groups):
    pools = []
    for group in groups:
        if isinstance(group, int):
            pools.append(rels[group])
            continue
        members = []
        for member in group:
            members.append(rels[member])
        pools.append(Maximum()(members))
    return Concatenate()(pools)
