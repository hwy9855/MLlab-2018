import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifiers = []
        self.weak_classifier_weights = []

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        train_size = len(y)
        w = np.ones(train_size)/train_size
        for iters in range(self.n_weakers_limit):
            clf = self.weak_classifier(max_depth = 3)
            # weaker the clf
            clf.fit(X, y, w)
            self.weak_classifiers.append(clf)
            y_pre = clf.predict(X)
            e = (w * (y_pre != y).astype(float)).sum()
            alpha = np.log((1 - e) / e) / 2
            self.weak_classifier_weights.append(alpha)
            w = w * np.exp(-alpha * y * y_pre)
            w = w / w.sum()


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        scores = []
        for i in range(self.n_weakers_limit):
            scores.append(self.weak_classifiers[i].predict(X) * self.weak_classifier_weights[i])
        return np.array(scores)

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y_pre = self.predict_scores(X)
        y_pre -= threshold
        y_pre[y_pre >= 0] = 1
        y_pre[y_pre < 0] = -1
        print(self.weak_classifier_weights)
        return y_pre[0]

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
