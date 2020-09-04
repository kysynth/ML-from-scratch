import heapq
import numpy as np

class SimpleKNeighborsClassifier():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        result = []
        for i in range(X_test.shape[0]):
            curr_X_test = X_test[i]
            min_heap = []
            for j in range(self.X_train.shape[0]):
                dis = self._l2_distance(curr_X_test, self.X_train[j])
                heapq.heappush(min_heap, (-dis, self.y_train[j]))
                if len(min_heap) > self.n_neighbors:
                    heapq.heappop(min_heap)

            votes = {}
            for j in range(len(min_heap)):
                _, vote_class = min_heap[j]
                if vote_class not in votes:
                    votes[vote_class] = 1
                else:
                    votes[vote_class] += 1

            max_vote = -1
            max_vote_class = None
            for vote_class, num_vote in votes.items():
                if num_vote > max_vote:
                    max_vote = num_vote
                    max_vote_class = vote_class

            result.append(np.array([max_vote_class]))
        return np.concatenate(result)

    def _l2_distance(self, a, b):
        diff = a - b
        return np.sqrt(np.sum(np.square(a - b)))
