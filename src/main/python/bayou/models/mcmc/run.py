"""
0) Train or fetch the model (Optional)
1) Prompt the user for API or type constraints
1.1) Display a list of legal API calls and types
2) Check if the given constraints are on legal evidences
3) Initialize the tree
4) Run MCMC for burnin_period iterations
5) Return the n+1th sketch
"""
import json
import sys
import argparse
import textwrap

from bayou.models.mcmc.init_tree import Tree
from bayou.models.mcmc.data_reader import Reader
from bayou.models.mcmc.utils import read_config

HELP = """\
Config options should be given as a JSON file (see config.json for example):
{                                         |
    "model": "lle"                        | The implementation id of this model (do not change)
    "latent_size": 32,                    | Latent dimensionality
    "batch_size": 50,                     | Minibatch size
    "num_epochs": 100,                    | Number of training epochs
    "learning_rate": 0.02,                | Learning rate
    "print_step": 1,                      | Print training output every given steps
    "evidence": [                         | Provide each evidence type in this list
        {                                 |
            "name": "apicalls",           | Name of evidence ("apicalls")
            "units": 64,                  | Size of the encoder hidden state
            "num_layers": 3               | Number of densely connected layers
            "tile": 1                     | Repeat the encoding n times (to boost its signal)
        },                                |
        {                                 |
            "name": "types",              | Name of evidence ("types")
            "units": 32,                  | Size of the encoder hidden state
            "num_layers": 3               | Number of densely connected layers
            "tile": 1                     | Repeat the encoding n times (to boost its signal)
        },                                |
        {                                 |
            "name": "keywords",           | Name of evidence ("keywords")
            "units": 64,                  | Size of the encoder hidden state
            "num_layers": 3               | Number of densely connected layers
            "tile": 1                     | Repeat the encoding n times (to boost its signal)
        }                                 |
    ],                                    |
    "decoder": {                          | Provide parameters for the decoder here
        "units": 256,                     | Size of the decoder hidden state
        "num_layers": 3,                  | Number of layers in the decoder
        "max_ast_depth": 32               | Maximum depth of the AST (length of the longest path)
    }
    "reverse_encoder": {
        "units": 256,
        "num_layers": 3,
        "max_ast_depth": 32
    }                                   |
}                                         |
"""


def run(clargs):
    """
    Takes in the user's evidence constraints and generates a sketch out of it.
    :param clargs: command-line arguments specifying the config and data
    :return: a tree that satisfies the user's evidence constraints
    """
    # Create a new Reader to fetch the vocabulary
    with open("save/config.json") as f:
        config = read_config(json.load(f))
    reader = Reader(clargs, config)

    # Prompt the user for constraints
    constraints = dict()
    while True:
        user_in = input('Please enter the evidence constraints you want in a JSON object where the key is the API call '
                        'or type and the value is true or false. To see the vocabulary, enter \'--vocab\'. To finish, '
                        'enter \'--end\'.\n')
        if user_in == "--vocab":
            for call in reader.api_calls:
                print(call)
        elif user_in == "--end":
            break
        else:
            # Check if the given evidences exist and if the values are booleans
            constraint = json.loads(user_in)
            bad_ev = False
            for ev in constraint.keys():
                if ev not in reader.api_calls:
                    print(str(ev) + " is not a valid piece of evidence this model recognizes.")
                    bad_ev = True
                    break
                elif not isinstance(constraint[ev], bool):
                    print(str(constraint[ev]) + " is not a valid boolean value.")
                    bad_ev = True
                    break

            if not bad_ev:
                constraints = {**constraints, **constraint}

    # Initialize our tree
    tree = Tree(constraints, config)
    print(tree.candidate)

    # Step through the space of valid trees for burnin_period iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='',
                        help='checkpoint model during training here')
    parser.add_argument('--config', type=str, default=None,
                        help='config file (see description above for help)')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='ignore config options and continue training model checkpointed here')
    clargs = parser.parse_args(['--config', 'save/config.json', 'data/data.json'])
    sys.setrecursionlimit(clargs.python_recursion_limit)
    if not clargs.config:
        parser.error('Provide at least one option: --config')
    run(clargs)

