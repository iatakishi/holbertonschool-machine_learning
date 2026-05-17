#!/usr/bin/env python3
""" random forest """
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """ random forest """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """ predict """
        all_preds = np.array(
            [pred(explanatory) for pred in self.numpy_preds]
        )
        return np.array([
            np.bincount(all_preds[:, i]).argmax()
            for i in range(explanatory.shape[0])
        ])

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """ fit """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(
                T.accuracy(T.explanatory, T.target)
            )
        if verbose == 1:
            print(
                f"  Training finished.\n"
                f"    - Mean depth                     : "
                f"{np.array(depths).mean()}\n"
                f"    - Mean number of nodes           : "
                f"{np.array(nodes).mean()}\n"
                f"    - Mean number of leaves          : "
                f"{np.array(leaves).mean()}\n"
                f"    - Mean accuracy on training data : "
                f"{np.array(accuracies).mean()}\n"
                f"    - Accuracy of the forest on td   : "
                f"{self.accuracy(self.explanatory, self.target)}"
            )

    def accuracy(self, test_explanatory, test_target):
        """ accuracy """
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
