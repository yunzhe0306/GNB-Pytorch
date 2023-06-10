from club import CLUB
from locb import LOCB
from cofiba import COFIBA
from sclub import SCLUB
from neuucb_ind import neuucb_ind
from neuucb_one import neuucb_one
from Neural_TS import Neural_TS
from meta_ban import meta_ban
from DynUCB import DynUCB
from EE_Net import Exploitation, Exploration, Decision_maker
import argparse
import numpy as np
import sys
import torch

from load_data import load_yelp_new, load_mnist_only, load_movielen_real_features_new, \
    load_movielen_real_features_MORE_USER, load_yelp_MORE_USERS, Bandit_Classification_Datasets

import time

from datetime import datetime
import sys


# Logger
# Recording console output
class Logger(object):
    def __init__(self, stdout):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.terminal = sys.stdout
        self.log = open("./Benchmark-logs/Multi_RUNS_logfile_" + dt_string + "_benchmarks_.log", "w")
        self.out = stdout
        print("date and time =", dt_string)

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        self.terminal.write(message)

    def flush(self):
        pass

sys.stdout = Logger(sys.stdout)

if __name__ == '__main__':
    torch.cuda.set_device(1)
    # algo_list = ['meta_ban', 'neuucb_one', 'ee-net', 'neuucb_ind', 'DynUCB', 'locb',  'club', 'sclub', 'cofiba']
    algo_list = ['ee-net', 'neuucb_one', 'DynUCB', 'locb',  'club', 'sclub', 'cofiba', 'meta_ban', 'neuucb_ind']
    # algo_list = ['ee-net', 'sclub', 'cofiba', 'meta_ban', 'neuucb_ind']
    # algo_list = ['neuucb_one']
    # algo_list = ['meta_ban']
    # algo_list = ['club']
    # algo_list = ['DynUCB']
    #
    # algo_list = ['Neural_TS', 'neuucb_one']
    # algo_list = ['ee-net']
    # algo_list = ['locb', 'DynUCB', 'neuucb_one', 'club']

    # -----------------------------------------------------------------------------
    NUM_Runs = 5
    data_sets_list = ['movie_real', 'yelp', 'mnist_only', 'shuttle', 'letter', 'pendigits']

    # --------------------------------------------
    for data in data_sets_list:
        for p_i in range(NUM_Runs):
            print("--- Current run: {}/{}".format(p_i + 1, NUM_Runs))
            for algo_i, algo_name in enumerate(algo_list):
                s_time = time.time()
                rec_time_sum = 0.0

                parser = argparse.ArgumentParser(description='Meta-Ban')
                parser.add_argument('--dataset', default=data, type=str,
                                    help='mnist_only, yelp, movie_real, shuttle, notmnist, pendigits')
                parser.add_argument('--method', default=algo_name, type=str,
                                    help='locb, club, sclub, cofiba, neuucb_one, neuucb_ind, meta_ban, ee-net')
                args = parser.parse_args()

                data = args.dataset

                if data == "mnist_only":
                    b = load_mnist_only()

                elif data == "yelp":
                    b = load_yelp_new()

                elif data == "movie_real":
                    b = load_movielen_real_features_new()
                elif data in ['shuttle', 'letter', 'pendigits']:
                    b = Bandit_Classification_Datasets(name=data)
                else:
                    print("dataset is not defined. --help")
                    sys.exit()

                method = args.method
                print(data, method)
                print("User num: ", b.num_user)

                if method == "club":
                    model = CLUB(nu=b.num_user, d=b.dim)

                elif method == "locb":
                    # model = LOCB(nu=b.num_user, d=b.dim, gamma=0.5, num_seeds=profile[2], delta=0.7, detect_cluster=0) # Yelp
                    model = LOCB(nu=b.num_user, d=b.dim, gamma=0.2, num_seeds=b.num_user // 7, delta=0.1, detect_cluster=0)

                elif method == "DynUCB":
                    model = DynUCB(nu=b.num_user, d=b.dim, num_seeds=b.num_user // 7, delta=0.1, detect_cluster=0, alpha=0.1)

                elif method == "sclub":
                    model = SCLUB(nu=b.num_user, d=b.dim)

                elif method == "cofiba":
                    model = COFIBA(num_users=b.num_user, d=b.dim, num_rounds=10000, L=b.n_arm)

                elif method == "neuucb_ind":
                    model = neuucb_ind(dim=b.dim, n=b.num_user, n_arm=b.n_arm, lr=0.001)

                elif method == "neuucb_one":
                    model = neuucb_one(b.dim, lamdba=0.001, nu=0.01, full_g_matrix_flag=False)

                elif method == "Neural_TS":
                    model = Neural_TS(b.dim, lamdba=0.001, nu=0.01, full_g_matrix_flag=True)

                elif method == "ee-net":
                    lr_1 = 0.001
                    lr_2 = 0.001
                    lr_3 = 0.001

                    f_1 = Exploitation(b.dim, b.n_arm, lr_1)
                    f_2 = Exploration(b.n_arm - 1, 100, lr_2)
                    f_3 = Decision_maker(2, 20, lr_3)

                elif method == "meta_ban":
                    if data == "mnist_only" or data == "letter" or data == "pendigits":
                        model = meta_ban(dim=b.dim, n=b.num_user, n_arm=b.n_arm, gamma=0.20, lr=0.0001, user_side=1)
                    else:
                        model = meta_ban(dim=b.dim, n=b.num_user, n_arm=b.n_arm, gamma=0.32, lr=0.0001, user_side=0)

                else:
                    print("method is not defined. --help")
                    sys.exit()

                print(data, method)

                regrets = []
                summ = 0
                print("Round; Regret; Regret/Round")
                for t in range(10000):
                    u, context, rwd = b.step()

                    # ----------------------------------------------------------------
                    if method == "ee-net":
                        this_rec_time_s = time.time()
                        '''exploitation score and embedded gradient'''
                        res1_list, gra_list = f_1.output_and_gradient(context)

                        '''exploration score'''
                        res2_list = f_2.output(gra_list)

                        '''build input for decision maker'''
                        new_context = np.concatenate((res1_list, res2_list), axis=1)

                        '''hybrid decision maker'''
                        if t < 500:
                            '''sample linear model'''
                            suml = res1_list + res2_list
                            arm_select = np.argmax(suml)
                        else:
                            '''neural model'''
                            arm_select = f_3.select(new_context)

                        this_rec_time_e = time.time()
                        rec_time_sum += (this_rec_time_e - this_rec_time_s)

                        '''reward'''
                        r_1 = rwd[arm_select]

                        f_1.update(context[arm_select], r_1)
                        f_1_score = res1_list[arm_select][0]

                        '''label for exploration network'''
                        r_2 = r_1 - f_1_score
                        f_2.update(gra_list[arm_select], r_2)

                        '''creat additional samples for exploration network'''
                        if r_1 == 0:
                            index = 0
                            for i in gra_list:
                                '''set small scores for un-selected arms if the selected arm is 0-reward'''
                                c = (1 / np.log(t + 10))
                                if index != arm_select:
                                    f_2.update(i, c)
                                index += 1

                        '''label for decision maker'''
                        r_3 = float(r_1)
                        f_3.update(new_context[arm_select], r_3)

                        '''training'''
                        if t < 1000:
                            if t % 10 == 0:
                                loss_1 = f_1.train(t)
                                loss_2 = f_2.train(t)
                                loss_3 = f_3.train(t)
                        else:
                            if t % 100 == 0:
                                loss_1 = f_1.train(t)
                                loss_2 = f_2.train(t)
                                loss_3 = f_3.train(t)

                        #
                        r = rwd[arm_select]
                        reg = np.max(rwd) - r
                        summ += reg
                        regrets.append(summ)

                    # ----------------------------------------------------------------
                    else:
                        #
                        this_rec_time_s = time.time()
                        if method == "neuucb_ind" or method == "neuucb_one" or method == "Neural_TS":
                            arm_select, f_res, ucb = model.recommend(u, context, t)
                        elif method == "meta_ban":
                            arm_select, g, f_res, ucb = model.recommend(u, context, t)
                        else:
                            arm_select = model.recommend(u, context, t)
                        this_rec_time_e = time.time()
                        rec_time_sum += (this_rec_time_e - this_rec_time_s)

                        #
                        r = rwd[arm_select]
                        reg = np.max(rwd) - r
                        summ += reg
                        regrets.append(summ)

                        #
                        if method == "club" or method == "locb" or method == "DynUCB":
                            model.store_info(i=u, x=context[arm_select], y=r, t=t)
                            model.update(i=u, t=t)
                        if method == "cofiba":
                            model.store_info(i=u, x=context[arm_select], y=r, t=t)
                            model.update_cluster(i=u, kk=arm_select, t=t)
                        if method == "sclub":
                            model.store_info(i=u, x=context[arm_select], y=r, t=t, r=r, br=1.0)
                            model.split(u, t)
                            model.merge(t)
                            model.num_clusters[t] = len(model.clusters)
                        if method == "neuucb_ind" or method == "neuucb_one" or method == "Neural_TS":
                            model.update(u, context[arm_select], r)
                            if t < 1000:
                                if t % 10 == 0:
                                    loss = model.train(u, t)
                            else:
                                if t % 100 == 0:
                                    loss = model.train(u, t)
                        if method == "meta_ban":
                            model.update(u, context[arm_select], r, g)
                            if t < 1000:
                                if t % 10 == 0:
                                    loss = model.train(u, t)
                            else:
                                if t % 100 == 0:
                                    loss = model.train(u, t)

                    # ----------------------------------------------------------------

                    if t % 50 == 0:
                        print('Algo_name: {}, t: {}, Regret sum: {:}, Regret avg: {:.4f}'
                              .format(algo_name, t, summ, summ / (t + 1)))
                        print("Overall time elapsed: ", time.time() - s_time)
                        print("Overall running time elapsed: ", rec_time_sum)
                print("Algo_name: ", algo_name, ", round:", t, "; ", ", regret:", summ)
                np.save("./benchmark_regrets/" + algo_name + "{}_Run_{}_out_{}_regret.npy".format(data, str(p_i), str(NUM_Runs)),
                        regrets)
