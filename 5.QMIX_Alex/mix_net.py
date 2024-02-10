import torch
import torch.nn as nn
import torch.nn.functional as F

class QMIX_Net(nn.Module):
    def __init__(self, args):
        super(QMIX_Net, self).__init__()
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.batch_size = args.batch_size
        self.qmix_hidden_dim = args.qmix_hidden_dim
        
        self.hyper_w1 = nn.Linear(self.state_shape, self.n_agents * self.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(self.state_shape, self.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.state_shape, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_shape, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1))



    def forward(self, q, s, name):
        # q.shape(batch_size, max_episode_len, N)
        # s.shape(batch_size, max_episode_len, state_shape)

        # print(name)

        cur_batch_size = self.batch_size

        # print('---shapes---')
        # print(q.shape)
        # print(s.shape)

        q = q.view(-1, 1, self.n_agents)
        s = s.reshape(-1, self.state_shape)

        w1 = torch.abs(self.hyper_w1(s))
        b1 = self.hyper_b1(s)
        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)

        q_hidden = F.elu(torch.bmm(q, w1) + b1)

        w2 = torch.abs(self.hyper_w2(s))
        b2 = self.hyper_b2(s)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(q_hidden, w2) + b2
        q_total = q_total.view(cur_batch_size, -1, 1)
        return q_total