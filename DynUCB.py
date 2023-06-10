from collections import defaultdict
import numpy as np
import random
import sys
import networkx as nx
import sklearn.metrics.pairwise as Kernel


class Cluster:
    def __init__(self, users, S, b, N):
        self.users = set(users)  # a list/array of users
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)


class DynUCB:
    # each user is an independent LinUCB
    def __init__(self, nu, d, num_seeds, delta, detect_cluster, alpha=0.1):
        self.S = {i: np.eye(d) for i in range(nu)}
        self.b = {i: np.zeros(d) for i in range(nu)}
        self.Sinv = {i: np.eye(d) for i in range(nu)}
        self.theta = {i: np.zeros(d) for i in range(nu)}
        self.users = range(nu)

        self.seeds = range(num_seeds)
        self.clusters = {}

        # Change to random allocation
        self.cluster_inds = {i: [] for i in range(nu)}
        c_2_u_dict = {i: [] for i in range(num_seeds)}
        for i in self.users:
            init_c = random.choice(self.seeds)
            self.cluster_inds[i].append(init_c)
            c_2_u_dict[init_c].append(i)
        for seed in self.seeds:
            self.clusters[seed] = Cluster(users=c_2_u_dict[seed], S=np.eye(d), b=np.zeros(d), N=1)

        self.N = np.zeros(nu)
        self.alpha = alpha
        self.results = []

        self.d = d
        self.n = nu
        self.selected_cluster = 0
        self.delta = delta
        self.if_d = detect_cluster

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        return np.argmax(np.dot(items, theta) + self.alpha * (np.matmul(items, Sinv) * items).sum(axis=1))

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)

        return Sinv, theta

    def recommend(self, i, items, t):
        cls = self.cluster_inds[i][0]
        assert len(self.cluster_inds[i]) == 1
        cluster = self.clusters[cls]
        res_arm = self._select_item_ucb(cluster.S, cluster.Sinv, cluster.theta, items, cluster.N, t)

        return res_arm

    # def _select_item_ucb(self, S, Sinv, theta, items, N, t):
    #     ucbs = np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis=1)
    #     res = max(ucbs)
    #     it = np.argmax(ucbs)
    #
    #     return (res, it)

    def store_info(self, i, x, y, t):

        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1

        self.Sinv[i], self.theta[i] = self._update_inverse(self.S[i], self.b[i], self.Sinv[i], x, self.N[i])

        for c in self.cluster_inds[i]:
            self.clusters[c].S += np.outer(x, x)
            self.clusters[c].b += y * x
            self.clusters[c].N += 1
            self.clusters[c].Sinv = np.linalg.inv(self.clusters[c].S)
            self.clusters[c].theta = np.matmul(self.clusters[c].Sinv, self.clusters[c].b)

    def update(self, i, t):
        # def _factT(m):
        #     if self.if_d:
        #         delta = self.delta / self.n
        #         nu = np.sqrt(2 * self.d * np.log(1 + t) + 2 * np.log(2 / delta)) + 1
        #         de = np.sqrt(1 + m / 4) * np.power(self.n, 1 / 3)
        #         return nu / de
        #     else:
        #         return np.sqrt((1 + np.log(1 + m)) / (1 + m))
        # if i in self.clusters[seed].users:
        #     diff = self.theta[i] - self.theta[seed]
        #         if np.linalg.norm(diff) > _factT(self.N[i]) + _factT(self.N[seed]):
        #             self.clusters[seed].users.remove(i)
        #             self.cluster_inds[i].remove(seed)
        #             self.clusters[seed].S = self.clusters[seed].S - self.S[i] + np.eye(self.d)
        #             self.clusters[seed].b = self.clusters[seed].b - self.b[i]
        #             self.clusters[seed].N = self.clusters[seed].N - self.N[i]
        #
        #     else:
        #         diff = self.theta[i] - self.theta[seed]
        #         if np.linalg.norm(diff) < _factT(self.N[i]) + _factT(self.N[seed]):
        #             self.clusters[seed].users.add(i)
        #             self.cluster_inds[i].append(seed)
        #             self.clusters[seed].S = self.clusters[seed].S + self.S[i] - np.eye(self.d)
        #             self.clusters[seed].b = self.clusters[seed].b + self.b[i]
        #             self.clusters[seed].N = self.clusters[seed].N + self.N[i]

        # --------------------------------------------------
        update_flag = False
        current_c = self.cluster_inds[i][0]
        min_distance, min_c = 1000, current_c
        for seed in self.seeds:
            diff = self.theta[i] - self.clusters[seed].theta
            this_norm = np.linalg.norm(diff)
            if this_norm < min_distance:
                min_distance = this_norm
                min_c = seed

        if current_c != min_c:
            update_flag = True
            #
            self.clusters[current_c].users.remove(i)
            self.cluster_inds[i].remove(current_c)
            self.clusters[current_c].S = self.clusters[current_c].S - self.S[i] + np.eye(self.d)
            self.clusters[current_c].b = self.clusters[current_c].b - self.b[i]
            self.clusters[current_c].N = self.clusters[current_c].N - self.N[i]
            #
            self.clusters[min_c].users.add(i)
            self.cluster_inds[i].append(min_c)
            self.clusters[min_c].S = self.clusters[min_c].S + self.S[i] - np.eye(self.d)
            self.clusters[min_c].b = self.clusters[min_c].b + self.b[i]
            self.clusters[min_c].N = self.clusters[min_c].N + self.N[i]

        # Update cluster parameters
        if update_flag:
            #
            self.clusters[current_c].Sinv, self.clusters[current_c].theta = \
                self._update_inverse(self.clusters[current_c].S, self.clusters[current_c].b,
                                     self.clusters[current_c].Sinv, None,
                                     self.clusters[current_c].N)
            #
            self.clusters[min_c].Sinv, self.clusters[min_c].theta = \
                self._update_inverse(self.clusters[min_c].S, self.clusters[min_c].b, self.clusters[min_c].Sinv, None,
                                     self.clusters[min_c].N)
