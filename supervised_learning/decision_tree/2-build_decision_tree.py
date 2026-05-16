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
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """ right child add prefix """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
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
        return t


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
        return f"leaf [value={self.value}]"


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
        return self.root.__str__()
