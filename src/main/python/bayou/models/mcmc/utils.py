# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import re
import tensorflow as tf
from tensorflow.python.client import device_lib
from itertools import chain
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

CONFIG_GENERAL = ['model', 'latent_size', 'batch_size', 'num_epochs',
                  'learning_rate', 'print_step', 'checkpoint_step']
CONFIG_ENCODER = ['name', 'units', 'num_layers', 'tile', 'max_depth', 'max_nums', 'ev_drop_prob', 'ev_call_drop_prob']
CONFIG_DECODER = ['units', 'num_layers', 'max_ast_depth']
CONFIG_REVERSE_ENCODER = ['units', 'num_layers', 'max_ast_depth']
CONFIG_INFER = ['vocab', 'vocab_size']


def get_var_list():
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')
    rev_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Reverse_Encoder')
    bayou_vars = decoder_vars + rev_encoder_vars
    var_dict = {'all_vars': all_vars,
                'decoder_vars': decoder_vars,
                'bayou_vars': bayou_vars,
                'rev_encoder_vars': rev_encoder_vars}
    return var_dict


def plot_probs(prob_vals, fig_name ="rankedProb.pdf", logx = False):
    plt.figure()
    plot_path = os.path.join(os.getcwd(),'generation')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plt.grid()
    plt.title("Probability With Ranks")
    if logx:
        plt.semilogx(prob_vals)
    else:
        plt.plot(prob_vals)
    plt.xlabel("Ranks->")
    plt.ylabel("Log Probabilities")
    plt.savefig(os.path.join(plot_path, fig_name), bbox_inches='tight')
    return


def length(tensor):
    elems = tf.sign(tf.reduce_max(tensor, axis=2))
    return tf.reduce_sum(elems, axis=1)


# split s based on camel case and lower everything (uses '#' for split)
def split_camel(s):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1#\2', s)  # UC followed by LC
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1#\2', s1)  # LC followed by UC
    split = s1.split('#')
    return [s.lower() for s in split]


def normalize_log_probs(probs):
    sum = -1*np.inf
    for prob in probs:
        sum = np.logaddexp(sum, prob)

    for i in range(len(probs)):
        probs[i] -= sum
    return probs


def rank_statistic(_rank, total, prev_hits, cutoff):
    cutoff = np.array(cutoff)
    hits = prev_hits + (_rank < cutoff)
    prctg = hits / total
    return hits, prctg


# Do not move these imports to the top, it will introduce a cyclic dependency
import bayou.models.mcmc.evidence


# convert JSON to config
def read_config(js, chars_vocab=False):
    config = argparse.Namespace()

    for attr in CONFIG_GENERAL:
        config.__setattr__(attr, js[attr])

    config.evidence = bayou.models.mcmc.evidence.Evidence.read_config(js['evidence'], chars_vocab)[:3]
    config.decoder = argparse.Namespace()
    config.reverse_encoder = argparse.Namespace()

    for attr in CONFIG_DECODER:
        config.decoder.__setattr__(attr, js['decoder'][attr])
    if chars_vocab:
        for attr in CONFIG_INFER:
            config.decoder.__setattr__(attr, js['decoder'][attr])
    # added two paragraph  of new code for reverse encoder
    for attr in CONFIG_REVERSE_ENCODER:
        config.reverse_encoder.__setattr__(attr, js['reverse_encoder'][attr])
    if chars_vocab:
        for attr in CONFIG_INFER:
            config.reverse_encoder.__setattr__(attr, js['reverse_encoder'][attr])
    return config


# convert config to JSON
def dump_config(config):
    js = {}

    for attr in CONFIG_GENERAL:
        js[attr] = config.__getattribute__(attr)

    js['evidence'] = [ev.dump_config() for ev in config.evidence]
    js['decoder'] = {attr: config.decoder.__getattribute__(attr) for attr in CONFIG_DECODER + CONFIG_INFER}
    # added code for reverse encoder
    js['reverse_encoder'] = {attr: config.reverse_encoder.__getattribute__(attr) for attr in
                             CONFIG_REVERSE_ENCODER + CONFIG_INFER}
    return js


def gather_calls(node):
    """
    Gathers all call nodes (recursively) in a given AST node

    :param node: the node to gather calls from
    :return: list of call nodes
    """

    if type(node) is list:
        return list(chain.from_iterable([gather_calls(n) for n in node]))
    node_type = node['node']
    if node_type == 'DSubTree':
        return gather_calls(node['_nodes'])
    elif node_type == 'DBranch':
        return gather_calls(node['_cond']) + gather_calls(node['_then']) + gather_calls(node['_else'])
    elif node_type == 'DExcept':
        return gather_calls(node['_try']) + gather_calls(node['_catch'])
    elif node_type == 'DLoop':
        return gather_calls(node['_cond']) + gather_calls(node['_body'])
    else:  # this node itself is a call
        return [node]
