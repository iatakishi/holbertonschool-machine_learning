#!/usr/bin/env python3
""" isolation random forest """
import numpy as np
Isolation_Random_Tree = (
    __import__('10-isolation_tree').Isolation_Random_Tree
)


class Isolation_Random_Forest():
    """ isolation random forest """
    def __init__(
            self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """ predict """
        predictions = np.array(
            [f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """ fit """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(
                f"  Training finished.\n"
                f"    - Mean depth                     : "
                f"{np.array(depths).mean()}\n"
                f"    - Mean number of nodes           : "
                f"{np.array(nodes).mean()}\n"
                f"    - Mean number of leaves          : "
                f"{np.array(leaves).mean()}"
            )

    def suspects(self, explanatory, n_suspects):
        """ returns n_suspects rows with smallest mean depth """
        depths = self.predict(explanatory)
        indices = np.argsort(depths)[:n_suspects]
        return explanatory[indices], depths[indices]
