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
import numpy as np
import tensorflow as tf

import argparse
import os
import sys
import simplejson as json
import textwrap

import time
import random

from itertools import chain
from bayou.models.mcmc.infer import BayesianPredictor
from bayou.models.mcmc.utils import read_config
from bayou.models.mcmc.data_reader import Reader
from bayou.models.mcmc.node import plot_path
from bayou.models.mcmc.evidence import Keywords

with open('sample_program.json') as file:
    program_dict = json.load(file)


def test(clargs):
    clargs.continue_from = None  # None

    with open(os.path.join(clargs.save, 'config.json')) as f:
        config = read_config(json.load(f), chars_vocab=True)

    # config.decoder.max_ast_depth = 1
    iWantRandom = True
    # if (iWantRandom):
    #     config.batch_size = 1
    # else:
    #     config.batch_size = 20

    print("Loading Bayou predictor...")
    predictor = BayesianPredictor(clargs, config)  # goes to infer.BayesianPredictor
    print("Bayou predictor loaded!")

    # list of dictionaries (each prog is a dict)
    programs = program_dict['programs']

    # breadth_first_search
    if (iWantRandom):
        path_head = predictor.random_search(programs)
        path = path_head.depth_first_search()

        randI = random.randint(0, 1000)
        dot = plot_path(randI, path, 1.0)
        # print(randI)
        # print(path)

    # BEAM SEARCH
    else:
        candies = predictor.beam_search(programs, topK=config.batch_size)
        for i, candy in enumerate(candies):
             path = candy.head.depth_first_search()
             prob = candy.log_probabilty
        #
             dot = plot_path(i,path, prob)
        #     print(path)
        #     # print()
        jsons = predictor.get_jsons_from_beam_search(programs, topK=config.batch_size)

        with open('asts/output_' + clargs.evidence + '.json', 'w') as f:
           json.dump({'programs': programs, 'asts': jsons}, fp=f, indent=2)

    return


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input programs we push through the vae')
    parser.add_argument('--save', type=str, required=True,
                        help='checkpoint model during training here')
    parser.add_argument('--output_file', type=str, default=None,
                        help='output file to print probabilities')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')

    clargs = parser.parse_args()

    if not os.path.exists(os.getcwd() + '/asts'):
        os.makedirs('asts')
    if not os.path.exists(os.getcwd() + '/plots'):
        os.makedirs('plots')

    sys.setrecursionlimit(clargs.python_recursion_limit)
    test(clargs)
