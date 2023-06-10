from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as hcluster
import time


class load_mnist_only:
    def __init__(self):
        self.n_arm = 10
        self.dim = 793
        self.act_dim = 784

        #  mnist
        batch_size = 1
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                                  transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)
        self.dataiter = iter(train_loader)
        self.num_user = 10

    def step(self):
        x, y = self.dataiter.next()
        d = x.numpy()[0]
        d = d.reshape(self.act_dim)
        target = y.item()
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a:a +
                   self.act_dim] = d
        rwd = np.zeros(self.n_arm)
        # print(target)
        rwd[target] = 1
        return target, X, rwd


from scipy.stats import norm


class load_movielen_real_features_new:
    def __init__(self, user_num=50, cluster_num=-1, clustering_coef=-1):
        # Fetch data
        self.m = np.load("./processed_data/MovieLens_backup/MovieLens_1000user_10000arms_scores.npy")
        self.U = np.load("./processed_data/MovieLens_backup/MovieLens_user_matrix.npy")
        self.I = np.load("./processed_data/MovieLens_backup/MovieLens_arm_matrix.npy")
        self.I = normalize(self.I, axis=1)

        self.total_user_num = self.U.shape[0]

        self.n_arm = 10
        self.dim = 10
        self.num_user = user_num
        kmeans = KMeans(n_clusters=self.num_user, random_state=0).fit(self.U)
        self.groups = kmeans.labels_

        # User set
        self.total_user_set = range(self.total_user_num)

        # Get the clustering of users with threshold
        if clustering_coef > 0:
            # Get the rep for each group
            group_rep = np.zeros([user_num, self.dim])
            g_2_user_dict = {}
            for g_i in range(user_num):
                u_indices = np.argwhere(self.groups == g_i).reshape(-1, )
                g_2_user_dict[g_i] = u_indices
                u_features = self.U[u_indices, :]
                group_rep[g_i, :] = np.mean(u_features, axis=0)

            # Get clusters
            # print(group_rep)
            g_clusters = hcluster.fclusterdata(group_rep, clustering_coef, criterion="distance")
            current_c_num = len(set(g_clusters))
            print("----- Clustering coef: {}, number of user groups: {}, number of clusters: {}"
                  .format(clustering_coef, user_num, current_c_num))
            print(g_clusters)

            # Get user indices based on clusters
            assert current_c_num >= cluster_num
            uni_c, count = np.unique(g_clusters, return_counts=True)
            count_sorted = np.argsort(-count)
            uni_c_trunc = uni_c[count_sorted][:cluster_num].tolist()
            print("Selected cluster num: {}, cluster groups: {}".format(len(uni_c_trunc), uni_c_trunc))
            selected_group = []
            for c_i in uni_c_trunc:
                this_groups = np.argwhere(g_clusters == c_i).reshape(-1, )
                for g_i in this_groups:
                    selected_group.append(g_i)
            print("Selected group num: {} / {}, selected groups: {}".format(len(selected_group), user_num,
                                                                            selected_group))

            #
            selected_users_list = []
            user_2_new_group_indices = {}
            g_2_new_indices = {}
            for g_i, g in enumerate(selected_group):
                g_2_new_indices[g] = g_i
                u_indices = g_2_user_dict[g]
                for u_i in u_indices:
                    selected_users_list.append(u_i)
                    user_2_new_group_indices[u_i] = g_i
            self.total_user_set = selected_users_list
            self.num_user = len(selected_group)

            #
            for g_i in range(user_num):
                if g_i not in g_2_new_indices:
                    g_2_new_indices[g_i] = -1

            #
            self.pos_index = defaultdict(list)
            self.neg_index = defaultdict(list)
            neg_count, pos_count = 0, 0
            for i in self.m:
                if i[2] > 0.4:
                    self.pos_index[g_2_new_indices[self.groups[int(i[0])]]].append((i[0], i[1], i[2]))
                    pos_count += 1
                else:
                    self.neg_index[g_2_new_indices[self.groups[int(i[0])]]].append((i[0], i[1], i[2]))
                    neg_count += 1
            print("Count: ", pos_count, neg_count)
            self.groups = user_2_new_group_indices

        # -----------------------------------------------
        else:
            self.pos_index = defaultdict(list)
            self.neg_index = defaultdict(list)
            neg_count, pos_count = 0, 0
            for i in self.m:
                # if i[2] == 1:
                if i[2] > 0.4:
                    self.pos_index[self.groups[int(i[0])]].append((i[0], i[1], i[2]))
                    pos_count += 1
                else:
                    self.neg_index[self.groups[int(i[0])]].append((i[0], i[1], i[2]))
                    neg_count += 1
            print("Count: ", pos_count, neg_count)

    def step(self):
        u = np.random.choice(self.total_user_set)

        g = self.groups[u]
        arm = np.random.choice(range(self.n_arm))
        # print(pos_index.shape)
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        #
        # pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), 9, replace=True)]
        # neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]
        # X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)

        # Positive prediction
        # pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), replace=False)]
        # neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), self.n_arm - 1, replace=False)]
        # X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        # ----- Negative prediction
        pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), self.n_arm - 1, replace=True)]
        neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)

        #
        X = []
        rwd_list = []
        for ind in X_ind:
            # X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.I[int(ind[1])])
            rwd_list.append(ind[2])

        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        # rwd = np.array(rwd_list)

        # contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        contexts = np.array(X)
        return g, contexts, rwd


# =====================================================================
class load_movielen_real_features_MORE_USER:
    def __init__(self, user_num=1000, cluster_num=-1, clustering_coef=-1):
        # Fetch data
        self.m = np.load("./processed_data/MovieLens_10000user_10000arms_scores.npy")
        self.U = np.load("./processed_data/MovieLens_user_matrix.npy")
        self.I = np.load("./processed_data/MovieLens_arm_matrix.npy")
        self.I = normalize(self.I, axis=1)

        self.total_user_num = self.U.shape[0]
        self.total_item_num = self.I.shape[0]
        print("---- Raw user num: ", self.total_user_num)
        print("---- Raw item num: ", self.total_item_num)
        print("--- Actual user num: ", user_num)

        self.n_arm = 10
        self.dim = 10
        self.num_user = user_num
        kmeans = KMeans(n_clusters=self.num_user, random_state=0).fit(self.U)
        self.groups = kmeans.labels_

        unique, counts = np.unique(self.groups, return_counts=True)
        print("--- Uniques: ", unique)
        print("--- Counts: ", counts)

        # User set
        self.total_user_set = range(self.total_user_num)

        # Get the clustering of users with threshold
        if clustering_coef > 0:
            # Get the rep for each group
            group_rep = np.zeros([user_num, self.dim])
            g_2_user_dict = {}
            for g_i in range(user_num):
                u_indices = np.argwhere(self.groups == g_i).reshape(-1, )
                g_2_user_dict[g_i] = u_indices
                u_features = self.U[u_indices, :]
                group_rep[g_i, :] = np.mean(u_features, axis=0)

            # Get clusters
            # print(group_rep)
            g_clusters = hcluster.fclusterdata(group_rep, clustering_coef, criterion="distance")
            current_c_num = len(set(g_clusters))
            print("----- Clustering coef: {}, number of user groups: {}, number of clusters: {}"
                  .format(clustering_coef, user_num, current_c_num))
            print(g_clusters)

            # Get user indices based on clusters
            assert current_c_num >= cluster_num
            uni_c, count = np.unique(g_clusters, return_counts=True)
            count_sorted = np.argsort(-count)
            uni_c_trunc = uni_c[count_sorted][:cluster_num].tolist()
            print("Selected cluster num: {}, cluster groups: {}".format(len(uni_c_trunc), uni_c_trunc))
            selected_group = []
            for c_i in uni_c_trunc:
                this_groups = np.argwhere(g_clusters == c_i).reshape(-1, )
                for g_i in this_groups:
                    selected_group.append(g_i)
            print("Selected group num: {} / {}, selected groups: {}".format(len(selected_group), user_num,
                                                                            selected_group))

            #
            selected_users_list = []
            user_2_new_group_indices = {}
            g_2_new_indices = {}
            for g_i, g in enumerate(selected_group):
                g_2_new_indices[g] = g_i
                u_indices = g_2_user_dict[g]
                for u_i in u_indices:
                    selected_users_list.append(u_i)
                    user_2_new_group_indices[u_i] = g_i
            self.total_user_set = selected_users_list
            self.num_user = len(selected_group)

            #
            for g_i in range(user_num):
                if g_i not in g_2_new_indices:
                    g_2_new_indices[g_i] = -1

            #
            self.pos_index = defaultdict(list)
            self.neg_index = defaultdict(list)
            neg_count, pos_count = 0, 0
            for i in self.m:
                if i[2] > 0.4:
                    self.pos_index[g_2_new_indices[self.groups[int(i[0])]]].append((i[0], i[1], i[2]))
                    pos_count += 1
                else:
                    self.neg_index[g_2_new_indices[self.groups[int(i[0])]]].append((i[0], i[1], i[2]))
                    neg_count += 1
            print("Count: ", pos_count, neg_count)
            self.groups = user_2_new_group_indices

        # -----------------------------------------------
        else:
            self.pos_index = defaultdict(list)
            self.neg_index = defaultdict(list)
            neg_count, pos_count = 0, 0
            for i in self.m:
                # if i[2] == 1:
                if i[2] > 0.4:
                    self.pos_index[self.groups[int(i[0])]].append((i[0], i[1], i[2]))
                    pos_count += 1
                else:
                    self.neg_index[self.groups[int(i[0])]].append((i[0], i[1], i[2]))
                    neg_count += 1
            print("Count: ", pos_count, neg_count)

    def step(self):
        u = np.random.choice(self.total_user_set)

        g = self.groups[u]
        arm = np.random.choice(range(self.n_arm))
        # print(pos_index.shape)
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        #
        # pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), 9, replace=True)]
        # neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]
        # X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)

        # Positive prediction
        # pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), replace=False)]
        # neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), self.n_arm - 1, replace=False)]
        # X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        # ----- Negative prediction
        if p_d >= self.n_arm - 1:
            pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), self.n_arm - 1, replace=False)]
        else:
            pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), self.n_arm - 1, replace=True)]
        neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)

        #
        X = []
        rwd_list = []
        for ind in X_ind:
            # X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.I[int(ind[1])])
            rwd_list.append(ind[2])

        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        # rwd = np.array(rwd_list)

        # contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        contexts = np.array(X)
        return g, contexts, rwd


class load_yelp_new:
    def __init__(self, user_num=50, cluster_num=-1, clustering_coef=-1):
        # Fetch data
        self.m = np.load("./processed_data/yelp_2000users_10000items_entry.npy")
        self.U = np.load("./processed_data/yelp_2000users_10000items_features.npy")
        self.I = np.load("./processed_data/yelp_10000items_2000users_features.npy")
        self.total_user_num = self.U.shape[0]
        self.total_item_num = self.I.shape[0]
        print("---- Raw user num: ", self.total_user_num)
        print("---- Raw item num: ", self.total_item_num)
        print("Dim: ", self.I.shape[1])
        kmeans = KMeans(n_clusters=user_num, random_state=0).fit(self.U)
        self.groups = kmeans.labels_
        self.n_arm = 10
        self.dim = 10
        self.num_user = user_num

        unique, counts = np.unique(self.groups, return_counts=True)
        print("--- Uniques: ", unique)
        print("--- Counts: ", counts)

        # User set
        self.total_user_set = range(self.total_user_num)

        # Get the clustering of users with threshold
        if clustering_coef > 0 and cluster_num > 0:
            # Get the rep for each group
            group_rep = np.zeros([user_num, self.dim])
            g_2_user_dict = {}
            for g_i in range(user_num):
                u_indices = np.argwhere(self.groups == g_i).reshape(-1, )
                g_2_user_dict[g_i] = u_indices
                u_features = self.U[u_indices, :]
                group_rep[g_i, :] = np.mean(u_features, axis=0)

            # Get clusters
            # print(group_rep)
            g_clusters = hcluster.fclusterdata(group_rep, clustering_coef, criterion="distance")
            current_c_num = len(set(g_clusters))
            print("----- Clustering coef: {}, number of user groups: {}, number of clusters: {}"
                  .format(clustering_coef, user_num, current_c_num))
            print(g_clusters)

            # Get user indices based on clusters
            assert current_c_num >= cluster_num
            uni_c, count = np.unique(g_clusters, return_counts=True)
            count_sorted = np.argsort(-count)
            uni_c_trunc = uni_c[count_sorted][:cluster_num].tolist()
            print("Selected cluster num: {}, cluster groups: {}".format(len(uni_c_trunc), uni_c_trunc))
            selected_group = []
            for c_i in uni_c_trunc:
                this_groups = np.argwhere(g_clusters == c_i).reshape(-1, )
                for g_i in this_groups:
                    selected_group.append(g_i)
            print("Selected group num: {} / {}, selected groups: {}".format(len(selected_group), user_num,
                                                                            selected_group))

            #
            selected_users_list = []
            user_2_new_group_indices = {}
            g_2_new_indices = {}
            for g_i, g in enumerate(selected_group):
                g_2_new_indices[g] = g_i
                u_indices = g_2_user_dict[g]
                for u_i in u_indices:
                    selected_users_list.append(u_i)
                    user_2_new_group_indices[u_i] = g_i
            self.total_user_set = selected_users_list
            self.num_user = len(selected_group)

            #
            for g_i in range(user_num):
                if g_i not in g_2_new_indices:
                    g_2_new_indices[g_i] = -1

            #
            self.pos_index = defaultdict(list)
            self.neg_index = defaultdict(list)
            neg_count, pos_count = 0, 0
            for i in self.m:
                if i[2] > 0.4:
                    self.pos_index[g_2_new_indices[self.groups[int(i[0])]]].append((i[0], i[1], i[2]))
                    pos_count += 1
                else:
                    self.neg_index[g_2_new_indices[self.groups[int(i[0])]]].append((i[0], i[1], i[2]))
                    neg_count += 1
            print("Count: ", pos_count, neg_count)
            self.groups = user_2_new_group_indices

        # -----------------------------------------------
        else:
            self.pos_index = defaultdict(list)
            self.neg_index = defaultdict(list)
            neg_count, pos_count = 0, 0
            for i in self.m:
                # if i[2] == 1:
                if i[2] > 0.4:
                    self.pos_index[self.groups[int(i[0])]].append((i[0], i[1], i[2]))
                    pos_count += 1
                else:
                    self.neg_index[self.groups[int(i[0])]].append((i[0], i[1], i[2]))
                    neg_count += 1
            print("Count: ", pos_count, neg_count)

        #
        count = 0
        for g_i in range(self.num_user):
            p_d = len(self.pos_index[g_i])
            n_d = len(self.neg_index[g_i])
            if  p_d == 0:
                count += 1

        print("=== invalid count: ", count)


    def step(self):
        u = np.random.choice(self.total_user_set)
        g = self.groups[u]
        arm = np.random.choice(range(self.n_arm))
        # print(pos_index.shape)
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        # Positive prediction
        pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), replace=False)]
        if n_d >= self.n_arm - 1:
            neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), self.n_arm - 1, replace=False)]
        else:
            neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), self.n_arm - 1, replace=True)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        # Negative prediction
        # pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), self.n_arm - 1, replace=True)]
        # neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]
        # X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)

        # ---------------------
        X = []
        for ind in X_ind:
            # X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.I[ind[1]])
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1

        # contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        contexts = np.array(X)
        contexts = normalize(contexts, axis=1)

        return g, contexts, rwd


# -------------------------------------------------------------------------------------------------------------

class load_yelp_MORE_USERS:
    def __init__(self, user_num=50, cluster_num=-1, clustering_coef=-1):
        # Fetch data
        self.m = np.load("./processed_data/Yelp_MORE_USERS/yelp_10000user_10000arms.npy")
        self.U = np.load("./processed_data/Yelp_MORE_USERS/Yelp_10000user_d20.npy")
        self.I = np.load("./processed_data/Yelp_MORE_USERS/Yelp_10000arm_d20.npy")
        self.total_user_num = self.U.shape[0]
        self.total_item_num = self.I.shape[0]
        print("---- Raw user num: ", self.total_user_num)
        print("---- Raw item num: ", self.total_item_num)
        print("Dim: ", self.I.shape[1])
        kmeans = KMeans(n_clusters=user_num, random_state=0).fit(self.U)
        self.groups = kmeans.labels_
        self.n_arm = 10
        self.dim = 10
        self.num_user = user_num

        unique, counts = np.unique(self.groups, return_counts=True)
        print("--- Uniques: ", unique)
        print("--- Counts: ", counts)

        # User set
        self.total_user_set = range(self.total_user_num)

        # Get the clustering of users with threshold
        if clustering_coef > 0 and cluster_num > 0:
            # Get the rep for each group
            group_rep = np.zeros([user_num, self.dim])
            g_2_user_dict = {}
            for g_i in range(user_num):
                u_indices = np.argwhere(self.groups == g_i).reshape(-1, )
                g_2_user_dict[g_i] = u_indices
                u_features = self.U[u_indices, :]
                group_rep[g_i, :] = np.mean(u_features, axis=0)

            # Get clusters
            # print(group_rep)
            g_clusters = hcluster.fclusterdata(group_rep, clustering_coef, criterion="distance")
            current_c_num = len(set(g_clusters))
            print("----- Clustering coef: {}, number of user groups: {}, number of clusters: {}"
                  .format(clustering_coef, user_num, current_c_num))
            print(g_clusters)

            # Get user indices based on clusters
            assert current_c_num >= cluster_num
            uni_c, count = np.unique(g_clusters, return_counts=True)
            count_sorted = np.argsort(-count)
            uni_c_trunc = uni_c[count_sorted][:cluster_num].tolist()
            print("Selected cluster num: {}, cluster groups: {}".format(len(uni_c_trunc), uni_c_trunc))
            selected_group = []
            for c_i in uni_c_trunc:
                this_groups = np.argwhere(g_clusters == c_i).reshape(-1, )
                for g_i in this_groups:
                    selected_group.append(g_i)
            print("Selected group num: {} / {}, selected groups: {}".format(len(selected_group), user_num,
                                                                            selected_group))

            #
            selected_users_list = []
            user_2_new_group_indices = {}
            g_2_new_indices = {}
            for g_i, g in enumerate(selected_group):
                g_2_new_indices[g] = g_i
                u_indices = g_2_user_dict[g]
                for u_i in u_indices:
                    selected_users_list.append(u_i)
                    user_2_new_group_indices[u_i] = g_i
            self.total_user_set = selected_users_list
            self.num_user = len(selected_group)

            #
            for g_i in range(user_num):
                if g_i not in g_2_new_indices:
                    g_2_new_indices[g_i] = -1

            #
            self.pos_index = defaultdict(list)
            self.neg_index = defaultdict(list)
            neg_count, pos_count = 0, 0
            for i in self.m:
                if i[2] > 0.4:
                    self.pos_index[g_2_new_indices[self.groups[int(i[0])]]].append((i[0], i[1], i[2]))
                    pos_count += 1
                else:
                    self.neg_index[g_2_new_indices[self.groups[int(i[0])]]].append((i[0], i[1], i[2]))
                    neg_count += 1
            print("Count: ", pos_count, neg_count)
            self.groups = user_2_new_group_indices

        # -----------------------------------------------
        else:
            self.pos_index = defaultdict(list)
            self.neg_index = defaultdict(list)
            neg_count, pos_count = 0, 0
            for i in self.m:
                # if i[2] == 1:
                if i[2] > 0.4:
                    self.pos_index[self.groups[int(i[0])]].append((i[0], i[1], i[2]))
                    pos_count += 1
                else:
                    self.neg_index[self.groups[int(i[0])]].append((i[0], i[1], i[2]))
                    neg_count += 1
            print("Count: ", pos_count, neg_count)

        #
        count = 0
        self.user_set = set()
        for g_i in range(self.num_user):
            p_d = len(self.pos_index[g_i])
            n_d = len(self.neg_index[g_i])
            if n_d < 5 or p_d == 0:
                count += 1
            else:
                self.user_set.add(g_i)

        print("=== invalid count: ", count)


    def step(self):
        g = None
        while g not in self.user_set:
            u = np.random.choice(self.total_user_set)
            g = self.groups[u]

        arm = np.random.choice(range(self.n_arm))
        # print(pos_index.shape)
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        # Positive prediction
        pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), replace=False)]
        if n_d >= self.n_arm - 1:
            neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), self.n_arm - 1, replace=False)]
        else:
            neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), self.n_arm - 1, replace=True)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        # Negative prediction
        # pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), self.n_arm - 1, replace=True)]
        # neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]
        # X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)

        # ---------------------
        X = []
        for ind in X_ind:
            # X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.I[ind[1]])
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1

        # contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        contexts = np.array(X)
        contexts = normalize(contexts, axis=1)

        return g, contexts, rwd


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


class Bandit_Classification_Datasets:
    def __init__(self, name):
        # Fetch data
        if name == 'pendigits':
            X, y = fetch_openml(data_id=32, return_X_y=True)
            X = X.to_numpy()
            y = y.to_numpy()

            #
            le = LabelEncoder()
            le.fit(y)
            y = le.transform(y)
            print("Unique lables: ", np.unique(y))

            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'letter':
            X, y = fetch_openml(data_id=6, return_X_y=True)
            X = X.to_numpy()
            y = y.to_numpy()

            #
            le = LabelEncoder()
            le.fit(y)
            y = le.transform(y)
            print("Unique lables: ", np.unique(y))

            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True)
            X = X.to_numpy()
            y = y.to_numpy()
            print("Unique lables: ", np.unique(y))

            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        else:
            raise RuntimeError('Dataset does not exist')
        # Shuffle data
        self.X, self.y = shuffle(X, y)
        # generate one_hot coding:
        self.y_arm = np.array(self.y).astype(np.int)
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = int(np.max(self.y_arm) / 2 + 1)
        self.dim = self.X.shape[1] + self.n_arm
        self.act_dim = self.X.shape[1]
        self.num_user = np.max(self.y_arm) + 1
        print("Data dim: ", self.dim)
        print("Number of arms: ", self.n_arm)
        print("Number of users: ", self.num_user)

    def step(self):
        if self.cursor > (len(self.X) - 1):
            self.cursor = 0

        x = self.X[self.cursor]
        y = self.y_arm[self.cursor]
        target = int(y.item() / 2.0)
        X_n = []
        for i in range(self.n_arm):
            front = np.zeros((1 * i))
            back = np.zeros((1 * (self.n_arm - i)))
            new_d = np.concatenate((front, x, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        self.cursor += 1

        X_n = normalize(X_n, axis=1)
        return y.item(), X_n, rwd
