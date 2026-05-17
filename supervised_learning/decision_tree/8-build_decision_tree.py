#!/usr/bin/env python3
""" decision tree """
import numpy as np


class Node:
    """ node """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ max depth below """
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """ count nodes below """
        left_count = self.left_child.count_nodes_below(only_leaves)
        right_count = self.right_child.count_nodes_below(only_leaves)
        return (0 if only_leaves else 1) + left_count + right_count

    def left_child_add_prefix(self, text):
        """ left child add prefix """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0].lstrip("-> ") + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """ right child add prefix """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0].lstrip("-> ") + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """ str """
        if self.is_root:
            t = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            t = f"node [feature={self.feature}, threshold={self.threshold}]"
        t += "\n" + self.left_child_add_prefix(self.left_child.__str__())
        t += self.right_child_add_prefix(self.right_child.__str__())
        return t.rstrip("\n")

    def get_leaves_below(self):
        """ get leaves below """
        return (self.left_child.get_leaves_below()
                + self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """ update bounds below """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if child == self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """ update indicator """

        def is_large_enough(x):
            return np.all(np.array([np.greater(x[:, key], self.lower[key])
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([np.less_equal(x[:, key], self.upper[key])
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """ pred """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """ leaf """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ max depth below """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ count nodes below """
        return 1

    def __str__(self):
        """ str """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """ get leaves below """
        return [self]

    def update_bounds_below(self):
        """ update bounds below """
        pass

    def pred(self, x):
        """ pred """
        return self.value


class Decision_Tree():
    """ decision tree """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ depth """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ count nodes """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ str """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """ get leaves """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ update bounds """
        self.root.update_bounds_below()

    def pred(self, x):
        """ pred """
        return self.root.pred(x)

    def update_predict(self):
        """ update predict """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([
            leaves[np.argmax(np.array([leaf.indicator(A)
                                       for leaf in leaves])[:, i])].value
            for i in range(A.shape[0])
        ])

    def fit(self, explanatory, target, verbose=0):
        """ fit """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                f"  Training finished.\n"
                f"    - Depth                     : {self.depth()}\n"
                f"    - Number of nodes           : {self.count_nodes()}\n"
                f"    - Number of leaves          : "
                f"{self.count_nodes(only_leaves=True)}\n"
                f"    - Accuracy on training data : "
                f"{self.accuracy(self.explanatory, self.target)}"
            )

    def np_extrema(self, arr):
        """ np extrema """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """ random split criterion """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """ fit node """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
                self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~(
                self.explanatory[:, node.feature] > node.threshold)

        is_left_leaf = (np.sum(left_population) < self.min_pop or
                        node.depth + 1 >= self.max_depth or
                        np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (np.sum(right_population) < self.min_pop or
                         node.depth + 1 >= self.max_depth or
                         np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """ get leaf child """
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ get node child """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """ accuracy """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def possible_thresholds(self, node, feature):
        """ possible thresholds """
        values = np.unique(
            (self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """ Gini split criterion one feature """
        thresholds = self.possible_thresholds(node, feature)
        feature_values = self.explanatory[
            node.sub_population, feature]
        classes = self.target[node.sub_population]
        n = len(classes)
        n_classes = np.max(classes) + 1

        Left_F = (
                         feature_values[:, np.newaxis] > thresholds[np.newaxis, :]
                 )[:, :, np.newaxis] & (
                         classes[:, np.newaxis, np.newaxis]
                         == np.arange(n_classes)[np.newaxis, np.newaxis, :]
                 )

        left_counts = Left_F.sum(axis=0)
        right_counts = (
                (classes[:, np.newaxis, np.newaxis]
                 == np.arange(n_classes)[np.newaxis, np.newaxis, :])
                .sum(axis=0) - left_counts
        )

        left_sizes = left_counts.sum(axis=1, keepdims=True)
        right_sizes = right_counts.sum(axis=1, keepdims=True)

        left_gini = 1 - np.sum(
            (left_counts / np.where(left_sizes == 0, 1, left_sizes)) ** 2,
            axis=1)
        right_gini = 1 - np.sum(
            (right_counts / np.where(right_sizes == 0, 1, right_sizes)) ** 2,
            axis=1)

        gini_avg = (
                           left_sizes[:, 0] * left_gini
                           + right_sizes[:, 0] * right_gini) / n

        best = np.argmin(gini_avg)
        return thresholds[best], gini_avg[best]

    def Gini_split_criterion(self, node):
        """ Gini split criterion """
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
