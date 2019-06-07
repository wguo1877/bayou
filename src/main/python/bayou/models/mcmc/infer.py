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
import tensorflow as tf
import numpy as np
from copy import deepcopy, copy

import os
import pickle
import json

from bayou.models.mcmc.model import Model
from bayou.models.mcmc.data_reader import Reader
from bayou.models.mcmc.architecture import BayesianReverseEncoder, BayesianDecoder
from bayou.models.mcmc.node import CHILD_EDGE, SIBLING_EDGE, Node
from bayou.models.mcmc.utils import read_config

MAX_GEN_UNTIL_STOP = 20
MAX_AST_DEPTH = 5

class TooLongPathError(Exception):
    pass


class IncompletePathError(Exception):
    pass


class InvalidSketchError(Exception):
    pass


class Candidate():
    def __init__(self, initial_state):
        self.tree_currNode = Node("DSubTree")
        self.head = self.tree_currNode

        self.last_item = self.tree_currNode.val
        self.last_edge = SIBLING_EDGE
        self.branch_stack = []

        self.length = 1
        self.log_probabilty = -np.inf
        self.state = initial_state

        self.rolling = True


class BayesianPredictor(object):

    def __init__(self, clargs, config):
        self.sess = tf.InteractiveSession()
        self.clargs = clargs
        self.config = config

        reader = Reader(clargs, config, dataIsThere=False)

        # Placeholders for tf data
        nodes_placeholder = tf.placeholder(reader.nodes.dtype, reader.nodes.shape)
        edges_placeholder = tf.placeholder(reader.edges.dtype, reader.edges.shape)
        targets_placeholder = tf.placeholder(reader.targets.dtype, reader.targets.shape)
        evidence_placeholder = [tf.placeholder(input.dtype, input.shape) for input in reader.inputs]

        # reset batches
        feed_dict = {fp: f for fp, f in zip(evidence_placeholder, reader.inputs)}
        feed_dict.update({nodes_placeholder: reader.nodes})
        feed_dict.update({edges_placeholder: reader.edges})
        feed_dict.update({targets_placeholder: reader.targets})

        dataset = tf.data.Dataset.from_tensor_slices((nodes_placeholder, edges_placeholder, targets_placeholder,
                                                      *evidence_placeholder))
        batched_dataset = dataset.batch(config.batch_size)
        iterator = batched_dataset.make_initializable_iterator()

        # initialize the iterator and get the data
        self.sess.run(iterator.initializer, feed_dict=feed_dict)
        newBatch = iterator.get_next()
        nodes, edges, targets = newBatch[:3]
        self.nodes = tf.transpose(nodes)
        self.edges = tf.transpose(edges)

        with tf.variable_scope("Reverse_Encoder"):
            embAPI = tf.get_variable('embAPI', [config.reverse_encoder.vocab_size, config.reverse_encoder.units])
            self.reverse_encoder = BayesianReverseEncoder(config, embAPI, self.nodes, self.edges)
            samples_1 = tf.random_normal([config.batch_size, config.latent_size], mean=0., stddev=1., dtype=tf.float32)

            # get a sample from the latent space
            self.psi_reverse_encoder = self.reverse_encoder.psi_mean + tf.sqrt(self.reverse_encoder.psi_covariance) * samples_1

        # setup the decoder with psi as the initial state
        with tf.variable_scope("Decoder"):
            emb = tf.get_variable('emb', [config.decoder.vocab_size, config.decoder.units])
            lift_w = tf.get_variable('lift_w', [config.latent_size, config.decoder.units])
            lift_b = tf.get_variable('lift_b', [config.decoder.units])

            self.initial_state = tf.nn.xw_plus_b(self.psi_reverse_encoder, lift_w, lift_b, name="Initial_State")
            self.decoder = BayesianDecoder(config, emb, self.initial_state, self.nodes, self.edges)

        with tf.name_scope("Loss"):
            output = tf.reshape(tf.concat(self.decoder.outputs, 1),
                                [-1, self.decoder.cell1.output_size])
            logits = tf.matmul(output, self.decoder.projection_w) + self.decoder.projection_b
            self.ln_probs = tf.nn.log_softmax(logits)
            self.idx = tf.multinomial(logits, 1)

            self.top_k_values, self.top_k_indices = tf.nn.top_k(self.ln_probs, k=config.batch_size)

        # restore the saved model
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(clargs.save)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def get_state(self, programs, num_psi_samples=1000):
        """
        Gets the initial state of the decoder by sampling from the reverse encoder (or directly from the latent space).
        :param programs: json file containing sample programs
        :param num_psi_samples:
        :return: initial state of decoder
        """
        # # read in data
        # reader = Reader(self.clargs, self.config, infer=True, dataIsThere=False)
        #
        # # create feed dict of reverse encoder inputs
        # nodes_placeholder = tf.placeholder(reader.nodes.dtype, reader.nodes.shape)
        # edges_placeholder = tf.placeholder(reader.edges.dtype, reader.edges.shape)
        # targets_placeholder = tf.placeholder(reader.targets.dtype, reader.targets.shape)
        # evidence_placeholder = [tf.placeholder(input.dtype, input.shape) for input in reader.inputs]
        #
        # feed_dict = {fp: f for fp, f in zip(evidence_placeholder, reader.inputs)}
        # feed_dict.update({nodes_placeholder: reader.nodes})
        # feed_dict.update({edges_placeholder: reader.edges})
        # feed_dict.update({targets_placeholder: reader.targets})

        # sequentially evaluate the reverse encoder psi and decoder initial state
        # psi = self.sess.run(self.psi_reverse_encoder, feed_dict)
        psi = self.sess.run(self.psi_reverse_encoder)
        print(np.array(psi).shape)
        feed = {self.psi_reverse_encoder: psi}
        state = self.sess.run(self.initial_state, feed)

        return state

    def beam_search(self, programs, topK):

        self.config.batch_size = topK

        init_state = self.get_state(programs, self.clargs, self.config)

        candies = [Candidate(init_state[0]) for k in range(topK)]
        candies[0].log_probabilty = -0.0

        i = 0
        while(True):
            # states was batch_size * LSTM_Decoder_state_size
            candies = self.get_next_output_with_fan_out(candies)
            #print([candy.head.breadth_first_search() for candy in candies])
            #print([candy.rolling for candy in candies])

            if self.check_for_all_STOP(candies): # branch_stack and last_item
                break

            i += 1

            if i == MAX_GEN_UNTIL_STOP:
                break

        candies.sort(key=lambda x: x.log_probabilty, reverse=True)

        return candies

    def check_for_all_STOP(self, candies):
        for candy in candies:
            if candy.rolling == True:
                return False

        return True

    def get_next_output_with_fan_out(self, candies):

        topK = len(candies)

        last_item = [[self.config.decoder.vocab[candy.last_item]] for candy in candies]
        last_edge = [[candy.last_edge] for candy in candies]
        states = [candy.state for candy in candies]

        feed = {}
        feed[self.nodes.name] = np.array(last_item, dtype=np.int32)
        feed[self.edges.name] = np.array(last_edge, dtype=np.bool)
        feed[self.initial_state.name] = np.array(states)

        [states, beam_ids, beam_ln_probs, top_idx] = self.sess.run([self.decoder.state, self.top_k_indices, self.top_k_values, self.idx] , feed)

        states = states[0]
        next_nodes = [[self.config.decoder.chars[idx] for idx in beam] for beam in beam_ids]


        # states is still topK * LSTM_Decoder_state_size
        # next_node is topK * topK
        # node_probs in  topK * topK
        # log_probabilty is topK

        log_probabilty = np.array([candy.log_probabilty for candy in candies])
        length = np.array([candy.length for candy in candies])

        for i in range(topK):
            if candies[i].rolling == False:
                length[i] = candies[i].length + 1
            else:
               length[i] = candies[i].length

        for i in range(topK): # denotes the candidate
            for j in range(topK): # denotes the items
                if candies[i].rolling == False and j > 0:
                   beam_ln_probs[i][j] = -np.inf
                elif candies[i].rolling == False and j == 0:
                   beam_ln_probs[i][j] = 0.0

        new_probs = log_probabilty[:,None]  + beam_ln_probs

        len_norm_probs = new_probs #/ np.power(length[:,None], 1.0)

        rows, cols = np.unravel_index(np.argsort(len_norm_probs, axis=None)[::-1], new_probs.shape)
        rows, cols = rows[:topK], cols[:topK]

        # rows mean which of the original candidate was finally selected
        new_candies = []
        for row, col in zip(rows, cols):
            new_candy = deepcopy(candies[row]) #candies[row].copy()
            if new_candy.rolling:
                new_candy.state = states[row]
                new_candy.log_probabilty = new_probs[row][col]
                new_candy.length += 1

                value2add = next_nodes[row][col]
                # print(value2add)


                if new_candy.last_edge == SIBLING_EDGE:
                    new_candy.tree_currNode = new_candy.tree_currNode.addAndProgressSiblingNode(Node(value2add))
                else:
                    new_candy.tree_currNode = new_candy.tree_currNode.addAndProgressChildNode(Node(value2add))


                # before updating the last item lets check for penultimate value
                if new_candy.last_edge == CHILD_EDGE and new_candy.last_item in ['DBranch', 'DExcept', 'DLoop']:
                     new_candy.branch_stack.append(new_candy.tree_currNode)
                     new_candy.last_edge = CHILD_EDGE
                     new_candy.last_item = value2add

                elif value2add in ['DBranch', 'DExcept', 'DLoop']:
                     new_candy.branch_stack.append(new_candy.tree_currNode)
                     new_candy.last_edge = CHILD_EDGE
                     new_candy.last_item = value2add

                elif value2add == 'STOP':
                     if len(new_candy.branch_stack) == 0:
                          new_candy.rolling = False
                     else:
                          new_candy.tree_currNode = new_candy.branch_stack.pop()
                          new_candy.last_item = new_candy.tree_currNode.val
                          new_candy.last_edge = SIBLING_EDGE
                else:
                     new_candy.last_edge = SIBLING_EDGE
                     new_candy.last_item = value2add

            new_candies.append(new_candy)

        return new_candies

    def get_jsons_from_beam_search(self, programs, topK):

        candidates = self.beam_search(programs, topK)

        candidates = [candidate for candidate in candidates if candidate.rolling is False]
        # candidates = candidates[0:1]
        # print(candidates[0].head.breadth_first_search())
        candidate_jsons = [self.paths_to_ast(candidate.head) for candidate in candidates]
        return candidate_jsons

    def paths_to_ast(self, head_node):
        """
        Converts a AST
        :param paths: the set of paths
        :return: the AST
        """
        json_nodes = []
        ast = {'node': 'DSubTree', '_nodes': json_nodes}
        self.expand_all_siblings_till_STOP(json_nodes, head_node.sibling)

        return ast

    def expand_all_siblings_till_STOP(self, json_nodes, head_node):
        """
        Updates the given list of AST nodes with those along the path starting from pathidx until STOP is reached.
        If a DBranch, DExcept or DLoop is seen midway when going through the path, recursively updates the respective
        node type.
        :param nodes: the list of AST nodes to update
        :param path: the path
        :param pathidx: index of path at which update should start
        :return: the index at which STOP was encountered if there were no recursive updates, otherwise -1
        """
        while head_node.val != 'STOP':
            node_value = head_node.val
            astnode = {}
            if node_value == 'DBranch':
                astnode['node'] = node_value
                astnode['_cond'] = []
                astnode['_then'] = []
                astnode['_else'] = []
                self.update_DBranch(astnode, head_node.child)
                json_nodes.append(astnode)
            elif node_value == 'DExcept':
                astnode['node'] = node_value
                astnode['_try'] = []
                astnode['_catch'] = []
                self.update_DExcept(astnode, head_node.child)
                json_nodes.append(astnode)
            elif node_value == 'DLoop':
                astnode['node'] = node_value
                astnode['_cond'] = []
                astnode['_body'] = []
                self.update_DLoop(astnode, head_node.child)
                json_nodes.append(astnode)
            else:
                json_nodes.append({'node': 'DAPICall', '_call': node_value})

            head_node = head_node.sibling

        return

    def update_DBranch(self, astnode, loop_node):
        """
        Updates a DBranch AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        # self.expand_all_siblings_till_STOP(astnode['_cond'], loop_node, pathidx+1)

        astnode['_cond'] = json_nodes = [{'node': 'DAPICall', '_call': loop_node.val}]
        self.expand_all_siblings_till_STOP(astnode['_then'], loop_node.sibling)
        self.expand_all_siblings_till_STOP(astnode['_else'], loop_node.child)
        return

    def update_DExcept(self, astnode, loop_node):
        """
        Updates a DExcept AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_try'], loop_node)
        self.expand_all_siblings_till_STOP(astnode['_catch'], loop_node.child)
        return

    def update_DLoop(self, astnode, loop_node):
        """
        Updates a DLoop AST node with nodes from the path starting at pathidx
        :param astnode: the AST node to update
        :param path: the path
        :param pathidx: index of path at which update should start
        """
        self.expand_all_siblings_till_STOP(astnode['_cond'], loop_node)
        self.expand_all_siblings_till_STOP(astnode['_body'], loop_node.child)
        return

    def get_encoder_mean_variance(self, evidences):
        # setup initial states and feed

        rdp = [ev.read_data_point(evidences, infer=True) for ev in self.config.evidence]
        inputs = [ev.wrangle([ev_rdp]) for ev, ev_rdp in zip(self.config.evidence, rdp)]

        feed = {}
        for j, ev in enumerate(self.config.evidence):
            feed[self.inputs[j].name] = inputs[j]

        [  encMean, encCovar ] = self.sess.run([ self.encoder.psi_mean , self.encoder.psi_covariance], feed)

        return encMean[0], encCovar[0]

    def random_search(self, programs):

        # got the state, to be used subsequently
        state = self.get_state(programs)
        print(state)
        start_node = Node("DSubTree")
        head, final_state = self.consume_siblings_until_STOP(state, start_node)

        return head.sibling

    def get_prediction(self, node, edge, state):
        feed = {}
        feed[self.nodes.name] = np.array([[self.config.decoder.vocab[node]]], dtype=np.int32)
        feed[self.edges.name] = np.array([[edge]], dtype=np.bool)
        feed[self.initial_state.name] = state

        [state,idx] = self.sess.run([self.decoder.state, self.idx] , feed)
        idx = idx[0][0]
        state = state[0]
        prediction = self.config.decoder.chars[idx]

        return Node(prediction), state

    def consume_siblings_until_STOP(self, state, init_node):
        # all the candidate solutions starting with a DSubTree node
        head = candidate = init_node
        if init_node.val == 'STOP':
            return head

        stack_QUEUE = []

        while True:
            predictionNode, state = self.get_prediction(candidate.val, SIBLING_EDGE, state)
            candidate = candidate.addAndProgressSiblingNode(predictionNode)

            prediction = predictionNode.val
            if prediction == 'DBranch':
                candidate.child, state = self.consume_DBranch(state)
            elif prediction == 'DExcept':
                candidate.child, state = self.consume_DExcept(state)
            elif prediction == 'DLoop':
                candidate.child, state = self.consume_DLoop(state)
            #end of inner while

            elif prediction == 'STOP':
                break

        #END OF WHILE
        return head, state

    def consume_DExcept(self, state):
        catchStartNode, state = self.get_prediction('DExcept', CHILD_EDGE, state)

        tryStartNode, state = self.get_prediction(catchStartNode.val, CHILD_EDGE, state)
        tryBranch , state = self.consume_siblings_until_STOP(state, tryStartNode)

        catchBranch, state = self.consume_siblings_until_STOP(state, catchStartNode)

        catchStartNode.child = tryStartNode

        return tryBranch, state

    def consume_DLoop(self, state):
        loopConditionNode, state = self.get_prediction('DLoop', CHILD_EDGE, state)
        loopStartNode, state = self.get_prediction(loopConditionNode.val, CHILD_EDGE, state)
        loopBranch, state = self.consume_siblings_until_STOP(state, loopStartNode)

        loopConditionNode.sibling = Node('STOP')
        loopConditionNode.child = loopBranch

        return loopConditionNode, state

    def consume_DBranch(self, state):
        ifStatementNode, state = self.get_prediction('DBranch', CHILD_EDGE, state)
        thenBranchStartNode, state = self.get_prediction(ifStatementNode.val, CHILD_EDGE, state)

        thenBranch , state = self.consume_siblings_until_STOP(state, thenBranchStartNode)
        ifElseBranch, state = self.consume_siblings_until_STOP(state, ifStatementNode)

        ifElseBranch.child = thenBranch

        return ifThenBranch, state
