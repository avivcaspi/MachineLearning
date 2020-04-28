import numpy as np
from scipy.stats import multivariate_normal
import copy


class EM:
    def __init__(self, n_components=1, max_iter=100):
        self.n_components = n_components
        self.params = dict()
        for i in range(self.n_components):
            self.params[i] = dict()
        self.features_num = 0
        self.params_results = []
        self.eps = 0
        self.max_iter = max_iter
        self.converged = False

    def fit(self, X):
        self.features_num = len(X[0])
        self.eps = 1e-6 * np.identity(self.features_num)
        num_samples = len(X)
        # initialize the parameters
        for i in range(self.n_components):
            # each component have features_num mean/cov/pi
            # Random initialize the mean of each feature, but within the values of the samples
            self.params[i]['mean'] = np.random.randint(np.min(X, axis=0), np.max(X, axis=0), size=self.features_num)
            # cov = nxn diagonal matrix
            self.params[i]['cov'] = np.diag(np.ones(self.features_num))
            # all pi are equals at the start
            self.params[i]['pi'] = 1 / self.n_components

        self.params_results.append(copy.deepcopy(self.params))
        last_log_likelihood = None
        log_likelihood = np.log(np.sum([self.params[c]['pi'] * multivariate_normal(self.params[c]['mean'], self.params[c]['cov']).pdf(X) for c in
                                                  range(self.n_components)]))
        iter = 0
        while last_log_likelihood is None or (log_likelihood > last_log_likelihood and iter < self.max_iter):
            iter += 1
            last_log_likelihood = log_likelihood
            # E step - for each sample we compute the probability it belongs to each component
            r = np.zeros((num_samples, self.n_components))

            for component in range(self.n_components):
                dist = multivariate_normal(mean=self.params[component]['mean'],
                                           cov=self.params[component]['cov'] + self.eps)
                x_prob = dist.pdf(X)    # probability of x belongs to component
                normalize_term = np.sum([self.params[i]['pi'] * multivariate_normal(mean=self.params[i]['mean'],
                                                                                    cov=self.params[i]['cov'] + self.eps).pdf(X) for i in range(self.n_components)], axis=0)
                r[:, component] = self.params[component]['pi'] * x_prob / normalize_term  # normalizing the probability

            # M step - for each component we update the parameters using the probability r we calculated
            mc = np.sum(r, axis=0)
            m = np.sum(mc)
            for component in range(self.n_components):
                self.params[component]['pi'] = mc[component] / m
                self.params[component]['mean'] = 1 / mc[component] * np.sum(r[:, component].reshape(-1, 1) * X, axis=0)
                self.params[component]['cov'] = 1 / mc[component] * \
                                                np.dot((r[:, component].reshape(-1, 1) * (
                                                            X - self.params[component]['mean'].reshape(1, -1))).T,
                                                       X - self.params[component]['mean'].reshape(1, -1)) + self.eps
            self.params_results.append(copy.deepcopy(self.params))
            log_likelihood = np.log(np.sum([self.params[c]['pi'] * multivariate_normal(self.params[c]['mean'], self.params[c]['cov']).pdf(X) for c in
                                                  range(self.n_components)]))

        if iter < self.max_iter:
            self.converged = True
        print(f'Converged = {self.converged}')
        return self.params_results

    # When given samples will predict the probability that each sample belongs to each component
    def predict(self, X):
        r = np.zeros((len(X), self.n_components))

        for component in range(self.n_components):
            dist = multivariate_normal(mean=self.params[component]['mean'],
                                       cov=self.params[component]['cov'] + self.eps)
            x_prob = dist.pdf(X)
            normalize_term = np.sum([multivariate_normal(mean=self.params[i]['mean'],
                                                         cov=self.params[i]['cov'] + self.eps).pdf(X)
                                     for i in range(self.n_components)], axis=0)
            r[:, component] = x_prob / normalize_term

        return r

    def means_(self):
        return [list(v) for c in range(self.n_components) for k, v in self.params[c].items() if k == 'mean']

    def covs_(self):
        return [v.tolist() for c in range(self.n_components) for k, v in self.params[c].items() if k == 'cov']
