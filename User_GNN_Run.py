import torch
from User_GNN_Model_user_graph_per_arm import User_GNN_Bandit_Per_Arm
from Parameters_Profile import get_GNB_parameters
import argparse
import numpy as np
import time
from datetime import datetime
import sys
# from User_GNN_packages import *

from load_data import load_yelp_new, load_mnist_only, load_movielen_real_features_new, \
    load_movielen_real_features_MORE_USER, load_yelp_MORE_USERS, Bandit_Classification_Datasets


# Logger
# Recording console output
class Logger(object):
    def __init__(self, stdout):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.terminal = sys.stdout
        self.log = open("./User_GNN-logs/Multi_Runs_User_GNN_logfile_" + dt_string + "_.log", "w")
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
    torch.cuda.set_device(0)

    # -----------------------------
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    ####################
    Num_Runs = 5
    dataset_list = ['movie_real', 'yelp', 'mnist_only', 'shuttle', 'letter', 'pendigits']
    ####################

    #
    for data in dataset_list:
        for p_i in range(Num_Runs):
            print("--- Current run: {}/{}".format(p_i+1, Num_Runs))

            parser = get_GNB_parameters(dataset=data)
            args = parser.parse_args()

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

            algo_name = 'GNB'
            model = User_GNN_Bandit_Per_Arm(dim=b.dim, user_n=b.num_user, arm_n=b.n_arm, k=args.k,
                                                GNN_lr=args.GNN_lr, user_lr=args.user_lr,
                                                bw_reward=args.bw_reward, bw_conf_b=args.bw_conf_b,
                                                batch_size=args.batch_size,
                                                GNN_pooling_step_size=args.GNN_pool_step_size,
                                                user_pooling_step_size=args.user_pool_step_size,
                                                arti_explore_constant=args.arti_explore_constant,
                                                num_layer=-1, explore_param=args.explore_param,
                                                separate_explore_GNN=args.separate_explore_GNN,
                                                train_every_user_model=args.train_every_user_model,
                                                device=device)
            print(data, algo_name, args.GNN_lr, args.user_lr, args.bw_reward, args.bw_conf_b, args.k, args.batch_size,
                  args.GNN_pool_step_size, args.user_pool_step_size, args.arti_explore_constant,
                  args.train_every_user_model, args.separate_explore_GNN)


            regrets = []
            summ = 0
            running_time_sum, rec_time_sum = 0.0, 0.0
            print("Round; Regret; Regret/Round")
            start_t = time.time()
            for t in range(10000):
                u, contexts, rwd = b.step()

                this_rec_time_s = time.time()
                # Update user graphs
                model.update_user_graphs(contexts=contexts, user_i=u)
                this_g_update_time = time.time()

                # Recommendation
                arm_select, user_g, point_est, whole_gradients = model.recommend(u, contexts, t)

                #
                running_time_sum += (time.time() - this_rec_time_s)
                rec_time_sum += (time.time() - this_g_update_time)

                #
                r = rwd[arm_select]
                GNN_residual_reward = r - point_est

                reg = np.max(rwd) - r
                summ += reg
                regrets.append(summ)

                # Create additional samples for exploration network -----------------------------
                # Add artificial exploration info when made false predictions
                if r == 0 and args.arti_explore_constant > 0:
                    model.update_artificial_explore_info(t, u, arm_select, whole_gradients)

                # Update model info ---------------------------------------------------------------------------------------
                model.update_info(u_selected=u, a_selected=arm_select, contexts=contexts, reward=r, GNN_gradient=user_g,
                                  GNN_residual_reward=GNN_residual_reward)

                #
                if t < 1000:
                    if t % 10 == 0:
                        u_exploit_loss, u_explore_loss = model.train_user_models(u=u)
                        GNN_exploit_loss, GNN_explore_loss = model.train_GNN_models()
                        print("Loss: ", u_exploit_loss, u_explore_loss, GNN_exploit_loss, GNN_explore_loss)
                else:
                    if t % 100 == 0:
                        u_exploit_loss, u_explore_loss = model.train_user_models(u=u)
                        GNN_exploit_loss, GNN_explore_loss = model.train_GNN_models()
                        print("Loss: ", u_exploit_loss, u_explore_loss, GNN_exploit_loss, GNN_explore_loss)

                # ----------------------------- Print user graph statistics -----------------------------------------
                if t % 10 == 0:
                    # Exploitation
                    np_mat_exloit_no_norm = model.user_exploitation_graph_dict[arm_select].cpu().numpy()
                    np_mat_exploit = model.exploit_adj_m_normalized.cpu().numpy()
                    powered_mat = np.linalg.matrix_power(np_mat_exploit, max(1, args.k))
                    off_diag_matrix = np.copy(powered_mat)
                    np.fill_diagonal(off_diag_matrix, 0)
                    off_diag_matrix_no_norm = np.copy(np_mat_exloit_no_norm)
                    np.fill_diagonal(off_diag_matrix_no_norm, 0)
                    print("Exploit - Un-normed raw Weight max:", np.max(np_mat_exloit_no_norm))
                    print("Exploit - Un-normed raw Weight avg:",
                          np.sum(np_mat_exloit_no_norm) / (np.count_nonzero(np_mat_exloit_no_norm)))
                    print("Exploit - Un-normed Off-diag Weight max:", np.max(off_diag_matrix_no_norm))
                    print("Exploit - Un-normed Off-diag Weight avg:",
                          np.sum(off_diag_matrix_no_norm) / (np.count_nonzero(off_diag_matrix_no_norm)))
                    print("Exploit - Normed Off-diag Weight max:", np.max(off_diag_matrix))
                    print("Exploit - Normed Off-diag Weight avg:",
                          np.sum(off_diag_matrix) / (np.count_nonzero(off_diag_matrix)))
                    print("Exploit - Powered Weight max:", np.max(powered_mat))
                    print("Exploit - Powered Weight avg:", np.sum(powered_mat) / (np.count_nonzero(powered_mat)))

                    # Exploration
                    np_mat_explore_no_norm = model.user_exploration_graph_dict[arm_select].cpu().numpy()
                    np_mat_explore = model.explore_adj_m_normalized.cpu().numpy()
                    powered_mat = np.linalg.matrix_power(np_mat_explore, max(1, args.k))
                    off_diag_matrix = np.copy(powered_mat)
                    np.fill_diagonal(off_diag_matrix, 0)
                    off_diag_matrix_no_norm = np.copy(np_mat_explore_no_norm)
                    np.fill_diagonal(off_diag_matrix_no_norm, 0)
                    print("Explore - Un-normed raw Weight max:", np.max(np_mat_explore_no_norm))
                    print("Explore - Un-normed raw Weight avg:",
                          np.sum(np_mat_explore_no_norm) / (np.count_nonzero(np_mat_explore_no_norm)))
                    print("Explore - Un-normed Off-diag Weight max:", np.max(off_diag_matrix_no_norm))
                    print("Explore - Un-normed Off-diag Weight avg:",
                          np.sum(off_diag_matrix_no_norm) / (np.count_nonzero(off_diag_matrix_no_norm)))
                    print("Explore - Normed Off-diag Weight max:", np.max(off_diag_matrix))
                    print("Explore - Normed Off-diag Weight avg:",
                          np.sum(off_diag_matrix) / (np.count_nonzero(off_diag_matrix)))
                    print("Explore - Normed Weight max:", np.max(powered_mat))
                    print("Explore - Normed Weight avg:", np.sum(powered_mat) / (np.count_nonzero(powered_mat)))

                if t == 300:
                    print()

                # ----------------------------- Print recommendation statistics -----------------------------
                if t % 20 == 0:
                    print('Algo_name: {}, t: {}, Regret sum: {:}, Regret avg: {:.4f}'
                          .format(algo_name, t, summ, summ / (t + 1)))
                    print("Overall elapsed: ", time.time() - start_t)
                    print("Running time elapsed: ", running_time_sum)
                    print("Recommendation time elapsed: ", rec_time_sum)
            print("Algo_name: ", algo_name, ", round:", t, "; ", ", regret:", summ)
            np.save("./User_GNN_regrets/" + algo_name + "{}_Run_{}_out_{}_regret.npy".format(data, str(p_i), str(Num_Runs)),
                    regrets)
