import numpy as np

class SimpleLinearRegression():
    def fit(self, X, y, method='closed'):
        n, p = X.shape
        X = np.concatenate([X, np.ones((n, 1))], axis=1)

        if method == 'closed':
            beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


        elif method == 'gd':
            beta = np.random.uniform(low=-1.0, high=1.0, size=(p + 1, ))
            learning_rate = 2e-3
            for i in range(100000):
                gradient = -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(beta)
                if i % 10000 == 0:
                    print("Iter %i: Obj: %.5f; Norm of gradient: %.5f" % (i, (y - X.dot(beta)).T.dot(y - X.dot(beta)), np.linalg.norm(gradient)))
                beta = beta - learning_rate * gradient

        # TODO: sgd / mini-batch

        self.coef_ = beta[:-1]
        self.intercept_ = beta[-1]

    def score(self, X, y):
        y_bar_vec = np.mean(y) * np.ones((y.shape))
        y_hat_vec = X.dot(self.coef_) + self.intercept_ * np.ones((y.shape))

        numerator = (y - y_hat_vec).T.dot(y - y_hat_vec)
        denominator = (y - y_bar_vec).T.dot(y - y_bar_vec)

        return 1 - numerator / denominator
