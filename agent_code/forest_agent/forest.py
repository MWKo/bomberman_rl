import numpy as np
from abc import ABCMeta, abstractmethod

class Node:
    '''
      this class will later get the following attributes
      all nodes:
          features
          responses
      split nodes additionally:
          left
          right
          split_index
          threshold
      leaf nodes additionally
          prediction
    '''
        

class Tree:
    '''
      base class for RegressionTree and ClassificationTree
    '''
    def __init__(self, n_min=10):
        '''n_min: minimum required number of instances in leaf nodes
        '''
        self.n_min = n_min 
    
    def predict(self, x):
        ''' return the prediction for the given 1-D feature vector x
        '''
        # first find the leaf containing the 1-D feature vector x
        node = self.root
        while not hasattr(node, "prediction"):
            j = node.split_index
            if x[j] <= node.threshold:
                node = node.left
            else:
                node = node.right
        # finally, return the leaf's prediction
        return node.prediction
        
    def train(self, features, responses, D_try=None):
        '''
        features: the feature matrix of the training set
        response: the vector of responses
        '''
        N, D = features.shape
        assert(responses.shape[0] == N)

        if D_try is None:
            D_try = int(np.sqrt(D)) # number of features to consider for each split decision
        
        # initialize the root node
        self.root = Node()
        self.root.features  = features
        self.root.responses = responses

        # build the tree
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            active_indices = self.select_active_indices(D, D_try)
            left, right = self.make_split_node(node, active_indices)
            if left is None: # no split found
                self.make_leaf_node(node)
            else:
                stack.append(left)
                stack.append(right)
    
    def make_split_node(self, node, indices):
        '''
        node: the node to be split
        indices: a numpy array of length 'D_try', containing the feature 
                         indices to be considered for the present split
                         
        return: None, None -- if no suitable split has been found, or
                left, right -- the children of the split
        '''
        # all responses equal => no improvement possible by any split
        if np.unique(node.responses).shape[0] == 1:
            return None, None
        
        # find best feature j_min (among 'indices') and best threshold t_min for the split
        l_min = float('inf')  # upper bound for the loss, later the loss of the best split
        j_min, t_min = None, None

        for j in indices:
            thresholds = self.find_thresholds(node, j)

            # compute loss for each threshold
            for t in thresholds:
                loss = self.compute_loss_for_split(node, j, t)

                # remember the best split so far 
                # (the condition is never True when loss = float('inf') )
                if loss < l_min:
                    l_min = loss
                    j_min = j
                    t_min = t

        if j_min is None: # no split found
            return None, None

        # create children for the best split
        left, right = self.make_children(node, j_min, t_min)

        # turn the current 'node' into a split node
        # (store children and split condition)
        node.left = left
        node.right = right
        node.split_index = j_min
        node.threshold = t_min
        
        # return the children (to be placed on the stack)
        return left, right
    
    def select_active_indices(self, D, D_try):
        ''' return a 1-D array with D_try randomly selected indices from 0...(D-1).
        '''
        return np.random.choice(D, D_try, replace=False)
        
    def find_thresholds(self, node, j):
        ''' return: a 1-D array with all possible thresholds along feature j
        '''

        # the thresholds are the averages between the sorted features
        sorted_features = np.sort(np.unique(node.features[:, j]))
        thresholds = (sorted_features[:-1] + sorted_features[1:]) / 2
        return thresholds

        
    def make_children(self, node, j, t):
        ''' execute the split in feature j at threshold t
        
            return: left, right -- the children of the split, with features and responses
                                   properly assigned according to the split
        '''

        left = Node()
        right = Node()

        at_most_t = node.features[:, j] <= t
        bigger_than_t = ~at_most_t
        
        left.features = node.features[at_most_t]
        left.responses = node.responses[at_most_t]
        right.features = node.features[bigger_than_t]
        right.responses = node.responses[bigger_than_t]
        
        return left, right
        
    @abstractmethod
    def make_leaf_node(self, node):
        ''' Turn node into a leaf by computing and setting `node.prediction`
        
            (must be implemented in a subclass)
        '''
        raise NotImplementedError("make_leaf_node() must be implemented in a subclass.")
        
    @abstractmethod
    def compute_loss_for_split(self, node, j, t):
        ''' Return the resulting loss when the data are split along feature j at threshold t.
            If the split is not admissible, return float('inf').
        
            (must be implemented in a subclass)
        '''
        raise NotImplementedError("compute_loss_for_split() must be implemented in a subclass.")

class RegressionTree(Tree):
    def __init__(self, n_min=10):
        super(RegressionTree, self).__init__(n_min)
        
    def compute_loss_for_split(self, node, j, t):
        # return the loss if we would split the instance along feature j at threshold t
        # or float('inf') if there is no feasible split
        
        at_most_t = node.features[:, j] <= t        
        y_left = node.responses[at_most_t]
        y_right = node.responses[~at_most_t]
        
        if y_left.shape[0] < self.n_min or y_right.shape[0] < self.n_min:
            return float('inf')
        
        loss = ((y_left - y_left.mean()) ** 2).sum() + ((y_right - y_right.mean()) ** 2).sum()
        return loss
        
    def make_leaf_node(self, node):
        # turn node into a leaf node by computing `node.prediction`
        # (note: the prediction of a regression tree is a real number)
        node.prediction = node.responses.mean()

def bootstrap_sampling(features, responses, label=None, balance_rest=False):
    """Return a bootstrap sample of features and responses."""
    if label is None:
        N = responses.shape[0]
        indices = np.random.choice(N, N, replace=True)

        return features[indices], responses[indices]
    # Sampling used for OVA classification.
    else:
        # Balance rest class. 
        if balance_rest == True:
            # Get minimum number of samples per class.
            num_samples = np.min([sum(responses == i) for i in range(10)])
            # Get number of samples per subclass or rest class.
            num_samples_rest_subclass = num_samples // 9

            # Sample from current digit/label.
            label_idx_samples = np.random.choice(np.where(responses == label)[0], size=num_samples, replace=True)
            # Sample from subclasses of rest class to build restclass bootstrap sample.
            rest_idx_samples = np.concatenate([np.random.choice(np.where(responses == i)[0], size=num_samples_rest_subclass, replace=True) for i in range(10) if i != label])

        # Do not balance rest class.
        else:
            ## Balance classes.
            # Get minimum number of samples.
            num_samples = min(sum(responses != label), sum(responses == label))
            # Sample from current digit/label.
            label_idx_samples = np.random.choice(np.where(responses == label)[0], size=num_samples, replace=True)
            # Sample from the all subclasses of rest class.
            rest_idx_samples = np.random.choice(np.where(responses != label)[0], size=num_samples, replace=True)

        # Combine the samples from the current digit and the "rest."
        indices = np.concatenate((label_idx_samples, rest_idx_samples))

        # OVA set of responses.
        _responses = responses.copy()
        # Set label of rest class.
        _responses[(_responses != label)] = -1

        return features[indices], _responses[indices]


class RegressionForest():
    def __init__(self, n_trees, n_min=10):
        # create ensemble
        self.trees = [RegressionTree(n_min) for i in range(n_trees)]
    
    def train(self, features, responses, label_specific=False, balance_rest=False):
        for i, tree in enumerate(self.trees):
            if label_specific == True:
                boostrap_features, bootstrap_responses = bootstrap_sampling(features, responses, i, balance_rest)
            else:
                boostrap_features, bootstrap_responses = bootstrap_sampling(features, responses)
            tree.train(boostrap_features, bootstrap_responses)

    def predict(self, x, pred_type="averaging"):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        # Compute the response of the ensemble by averaging.
        if pred_type == "averaging":
            return predictions.mean(axis=0)
        # Compute the response of the ensemble by choosing the class whose classifier
        # had the highest score.
        else:
            return np.argmax(predictions) if np.argmax(predictions) >= 0 else "unknown"