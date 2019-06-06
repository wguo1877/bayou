from bayou.models.mcmc.utils import CHILD_EDGE, SIBLING_EDGE
import collections


class Tree:
    """
    Given the constraints the user specifies, this class will generate a tree that matches them. Since each constraint
    implies that a given piece of evidence must be in or not in the final program, here we focus on generating a tree
    that has all the evidences the user wants and focus on the evidences the user does not want in the MCMC steps.
    """
    def __init__(self, constraints, config):
        self.constraints = constraints
        self.config = config

        # Create a node for each piece of evidence the user wants; each node is linked together by a child edge
        self.candidate = [('DSubTree', CHILD_EDGE)]
        for ev in self.constraints:
            # Skip all evidences that cannot appear in the program
            if not self.constraints[ev]:
                continue

            # Otherwise, create a node and add it to the candidate
            ast = collections.OrderedDict()

            # For now, we are only constraining upon API calls
            ast['node'] = 'DAPICall'
            ast['_call'] = ev

            self.candidate.append((ast, CHILD_EDGE))

    def generate_tree(self):
        """
        Generates a simple AST that fits the user's constraints.
        """


