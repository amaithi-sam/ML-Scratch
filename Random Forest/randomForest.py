import numpy as np
# from ..DecisionTree.decision_tree import DecisionTree
from decision_tree import DecisionTree

import numpy as np 
from collections import Counter

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True) # range of n_samples with size of n_samples having multiple(random) times of same indexes
    return X[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y) 
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest:
    
    def __init__(self, n_trees = 100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees 
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth 
        self.n_feats = n_feats  
        self.trees = []
    
    
    def fit(self, X, y):
        self.trees = []
        
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth,
                                n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        #[1111 0000 1111]
        #[101 101 101 101]          # 02:56:02 for explanation
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
        
        
        