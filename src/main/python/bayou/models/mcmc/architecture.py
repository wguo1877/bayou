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

import tensorflow as tf
from itertools import chain

from bayou.models.mcmc.gru_tree import TreeEncoder


class BayesianReverseEncoder(object):
    def __init__(self, config, emb, nodes, edges):
        self.inputs = [ev.placeholder(config) for ev in config.evidence]

        nodes = [ nodes[ config.reverse_encoder.max_ast_depth - 1 - i ] for i in range(config.reverse_encoder.max_ast_depth)]
        edges = [ edges[ config.reverse_encoder.max_ast_depth - 1 - i ] for i in range(config.reverse_encoder.max_ast_depth)]

        # Two halves: one set of NN for calculating the covariance and the other set for calculating the mean
        with tf.variable_scope("Covariance"):
            with tf.variable_scope("APITree"):
                API_Cov_Tree = TreeEncoder(emb, config.batch_size, nodes, edges, config.reverse_encoder.num_layers,
                                    config.reverse_encoder.units, config.reverse_encoder.max_ast_depth, config.latent_size)
                Tree_Cov = API_Cov_Tree.last_output

        with tf.variable_scope("Mean"):
            with tf.variable_scope('APITree'):
                API_Mean_Tree = TreeEncoder(emb, config.batch_size, nodes, edges, config.reverse_encoder.num_layers,
                                    config.reverse_encoder.units, config.reverse_encoder.max_ast_depth, config.latent_size)
                Tree_mean = API_Mean_Tree.last_output

            sigmas = Tree_Cov

            #dimension is  3*batch * 1
            # finalSigma = tf.layers.dense(tf.reshape( tf.transpose(tf.stack(sigmas, axis=0), perm=[1,0,2]), [config.batch_size, -1])  , config.latent_size, activation=tf.nn.tanh)
            finalSigma = tf.layers.dense(tf.reshape(sigmas, [config.batch_size, -1])  , config.latent_size, activation=tf.nn.tanh)
            finalSigma = tf.layers.dense(finalSigma, config.latent_size, activation=tf.nn.tanh)
            finalSigma = tf.layers.dense(finalSigma, 1)

            d = tf.tile(tf.square(finalSigma),[1, config.latent_size])
            d = .00000001 + d
            #denom = d # tf.tile(tf.reshape(d, [-1, 1]), [1, config.latent_size])
            #I = tf.ones([config.batch_size, config.latent_size], dtype=tf.float32)
            self.psi_covariance = d #I / denom

            encodings = Tree_mean

            # finalMean = tf.layers.dense(tf.reshape(tf.transpose(tf.stack(encodings, axis=0), perm=[1,0,2]), [config.batch_size, -1]) , config.latent_size, activation=tf.nn.tanh)
            finalMean = tf.layers.dense(tf.reshape(encodings, [config.batch_size, -1]) , config.latent_size, activation=tf.nn.tanh)
            finalMean = tf.layers.dense(finalMean, config.latent_size, activation=tf.nn.tanh)
            finalMean = tf.layers.dense(finalMean, config.latent_size)
            # 4. compute the mean of non-zero encodings
            self.psi_mean = finalMean


class BayesianDecoder(object):
    def __init__(self, config, emb, initial_state, nodes, edges):

        cells1, cells2 = [], []
        for _ in range(config.decoder.num_layers):
            cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units))
            cells2.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units))

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        # placeholders
        self.initial_state = [initial_state] * config.decoder.num_layers
        self.nodes = [nodes[i] for i in range(config.decoder.max_ast_depth)]
        self.edges = [edges[i] for i in range(config.decoder.max_ast_depth)]

        # projection matrices for output
        with tf.variable_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [self.cell1.output_size,
                                                                 config.decoder.vocab_size])
            self.projection_b = tf.get_variable('projection_b', [config.decoder.vocab_size])
            # tf.summary.histogram("projection_w", self.projection_w)
            # tf.summary.histogram("projection_b", self.projection_b)

        # setup embedding
        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in self.nodes)

        with tf.variable_scope('decoder_network'):
            # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
            with tf.variable_scope('rnn'):
                self.state = self.initial_state
                self.outputs = []
                # self.states = []
                for i, inp in enumerate(emb_inp):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('cell1'):  # handles CHILD_EDGE
                        output1, state1 = self.cell1(inp, self.state)
                    with tf.variable_scope('cell2'):  # handles SIBLING_EDGE
                        output2, state2 = self.cell2(inp, self.state)

                    output = tf.where(self.edges[i], output1, output2)
                    self.state = [tf.where(self.edges[i], state1[j], state2[j])
                                  for j in range(config.decoder.num_layers)]
                    self.outputs.append(output)
