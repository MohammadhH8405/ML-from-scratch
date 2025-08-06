'''
In this python script i will implement the differet ensemble learning methods
sush as bagging, random forest, boosting, adboosting and also make a decision tree and decision stump.
'''
import numpy as np 

class Decision_stump :
    '''
    A decision stump is a simple decision tree of depth 1.
    the decusion stump is usefull in ths and other scripts as a weak learner.
    '''
    def __init__(self) :
        self.feature = None
        self.threshold = None
        self.bigthresh = None
        self.smallthresh = None

    def _entropy(self, y, weights):
        '''
        This internaly used function calculates the entropy of the array given to it
        entropy is a measure of the uncertainty or randomness associated with a set. 
        Compute weighted class proportions for boosting if provided with
        '''
        total_weight = np.sum(weights)
        weighted_counts = np.bincount(y, weights=weights, minlength=np.max(y)+1)
        probs = weighted_counts / total_weight
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def build(self, x, y, weights=None):
        """
        Fit the decision stump by selecting the best feature and threshold
        that maximizes information gain.
        Args:
        x (array): Feature matrix of shape (n_samples, n_features).
        y (array): Target labels.
        weights (array, optional): Sample weights (used in boosting).
        """
        if weights is None:
            weights = np.ones(len(x))
        best_gain = -float('inf')
        # Try every feature and threshold to find the best split
        for feature_ind in range(x.shape[1]) :
            xmin, xmax = x[:,feature_ind].min(), x[:, feature_ind].max()
            for threshold in np.linspace(xmin, xmax, num=100) :
                # Binary split: smaller (<= threshold), bigger (> threshold)
                smaller = [threshold>=i for i in x[:,feature_ind]]
                bigger = [not i for i in smaller]
                # We have to check if they are empty since entropy doesn't make sence for no points
                if len(smaller) and len(bigger):
                    y_left, y_right = y[smaller], y[bigger]
                    w_left, w_right = weights[smaller], weights[bigger]
                    total_entropy = self._entropy(y, weights)
                    left_entropy = self._entropy(y_left, w_left)
                    right_entropy = self._entropy(y_right, w_right)
                    w_left_sum = np.sum(w_left)
                    w_right_sum = np.sum(w_right)
                    total_weight = w_left_sum + w_right_sum
                    weighted_entropy = (
                        (w_left_sum / total_weight) * left_entropy +
                        (w_right_sum / total_weight) * right_entropy)
                    gain = total_entropy - weighted_entropy
                    if gain > best_gain :
                        best_gain = gain
                        self.feature = feature_ind
                        self.threshold = threshold
                        self.bigthresh = np.bincount(y_right).argmax()
                        self.smallthresh = np.bincount(y_left).argmax()

    def predict(self, x):
        # Predicts and return an int or array of predictions of a given array x.
        if x.ndim == 1 : # Single sample (1D array)
            return self.bigthresh if x[self.feature] >= self.threshold else self.smallthresh
        else : # Multiple samples (2D array)
            feature_col = x[:, self.feature]
            return np.where(feature_col >= self.threshold, self.bigthresh, self.smallthresh)

class Bagging_classifier :
    # Ensemble classifier using Bagging with Decision Stumps.
    def __init__(self) :
        self.trees = None
        self.n_trees = None

    def _bootstrap_samples(self, x, y) :
        # Generate bootstrap samples (sample with replacement).
        n_samples = len(x)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return x[indices], y[indices]

    def build(self, x, y, n_trees=7) :
        """
        Train multiple decision stumps on bootstrap samples.
        Args:
        x (array): Feature matrix.
        y (array): Target labels.
        n_trees (int): Number of trees in the ensemble.
        """
        self.trees = []
        self.n_trees = n_trees
        for i in range(n_trees) :
            xsample, ysample = self._bootstrap_samples(x, y)
            tree = Decision_stump()
            tree.build(xsample, ysample)
            self.trees.append(tree)

    def predict(self, x):
        # Predict using majority voting from all trees.
        predictions = []
        for i in x:
            votes = [tree.predict(i) for tree in self.trees]
            predictions.append(np.bincount(votes).argmax())
        return np.array(predictions)

class Random_forest(Bagging_classifier) :
    # Random Forest classifier using feature subsetting on top of Bagging.
    def __init__(self) :
        super().__init__()

    def build(self, x, y, n_trees=7) :
        """
        Train trees with random feature subsets.
        Args:
        x (array): Feature matrix.
        y (array): Target labels.
        n_trees (int): Number of trees in the ensemble.
        """
        self.trees = []
        self.n_trees = n_trees
        n_features = x.shape[1]
        for i in range(n_trees) :
            xsample, ysample = self._bootstrap_samples(x, y)
            # Select a random subset of features (sqrt of total)
            k = int(np.sqrt(n_features))
            feature_indices = np.random.choice(n_features, size=k, replace=False)
            xsample_reduced = xsample[:, feature_indices]
            tree = Decision_stump()
            tree.build(xsample_reduced, ysample)
            self.trees.append((tree, feature_indices))

    def predict(self, x):
        # Predict using majority voting from trees trained on feature subsets.
        predictions = []
        for i in x:
            votes = []
            for tree, features in self.trees:
                vote = tree.predict(i[features])
                votes.append(vote)
            predictions.append(np.bincount(votes).argmax())
        return np.array(predictions)

class Boosting_classifier :
    '''
    Boosting classifier using AdaBoost-like algorithm with Decision Stumps.
    can predict data with multiple classes
    '''
    def __init__(self) :
        self.trees = None

    def build(self, x, y, n_trees=7) :
        # Train a sequence of decision stumps, adjusting weights after each one.
        weights = np.ones(len(x))
        self.trees = []
        for i in range(n_trees) :
            tree = Decision_stump()
            tree.build(x, y, weights=weights)
            predictions = tree.predict(x)
            # Calculate error
            error = np.sum(weights * (predictions != y))
            # Calculate the alpha (the weight of the learner)
            tree_weight = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.trees.append((tree, tree_weight))
            # Update weights: increase for wrong predictions
            weights *= np.exp(tree_weight * (predictions != y).astype(float).reshape(-1, 1))
            weights /= np.sum(weights)

    def predict(self, x):
        # Predict by weighted majority voting from all weak learners.
        final_predictions = []
        for i in x:
            votes = {}
            for tree, w in self.trees:
                pred = tree.predict(i)
                votes[pred] = votes.get(pred, 0) + w
            final_predictions.append(max(votes, key=votes.get))
        return np.array(final_predictions)
