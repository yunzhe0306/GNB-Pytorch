from packages import *

# torch.set_num_threads(8)
# torch.set_num_interop_threads(8)

arg_size = 1
arg_shuffle = 1
arg_seed = 0
arg_nu = 1
arg_lambda = 0.0001
arg_hidden = 100

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class Neural_TS:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100, full_g_matrix_flag=False):
        print("---Full matrix: ", full_g_matrix_flag)
        self.func = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        if full_g_matrix_flag:
            self.U = lamdba * torch.eye(self.total_param).to(device)
        else:
            self.U = lamdba * torch.ones((self.total_param,)).to(device)
        self.nu = nu

        #
        self.full_g_matrix_flag = full_g_matrix_flag

    def recommend(self, u, context, t):
        tensor = torch.from_numpy(context).float().to(device)
        mu = self.func(tensor)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        f_res = []
        ucb = []
        if self.full_g_matrix_flag:
            U_inv = torch.inverse(self.U)
        for fx in mu:
            # print("fx:", fx)
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            # print(self.lamdba)
            if self.full_g_matrix_flag:
                g_row = g.reshape([1, -1])
                sigma2 = self.nu * torch.matmul(g_row, torch.matmul(U_inv, g_row.T)).reshape(-1, )
                sigma = torch.sqrt(torch.sum(sigma2))
            else:
                sigma2 = self.lamdba * self.nu * g * g / self.U
                sigma = torch.sqrt(torch.sum(sigma2))
            f_res.append(fx.item())
            ucb.append(sigma.item())
            sample_r = torch.normal(mean=fx.item(), std=sigma)
            sampled.append(sample_r.item())
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled)
        if self.full_g_matrix_flag:
            g_array = g_list[arm].reshape(-1, )
            g_outer = torch.outer(g_array, g_array)
            self.U += g_outer
        else:
            self.U += g_list[arm] * g_list[arm]
        return arm, f_res, ucb

    def update(self, u, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

    def train(self, u, t):
        optimizer = optim.SGD(self.func.parameters(), lr=1e-3)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.to(device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length
