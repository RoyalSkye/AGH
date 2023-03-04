#!/usr/bin/env python

import math
import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, c_dim=4, x_dim=15, t_dim=2, k_f_dim=12, emb_dim=64, hidden=128, dropout=0.5):
        super(GCN, self).__init__()

        self.c_dim = c_dim
        self.x_dim = x_dim
        self.t_dim = t_dim
        self.k_f_dim = k_f_dim
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.c_embedding = nn.Linear(c_dim, emb_dim)
        self.x_embedding = nn.Linear(x_dim, emb_dim)
        self.t_embedding = nn.Linear(t_dim, emb_dim)
        self.k_f_embedding = nn.Linear(k_f_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, emb_dim)
        self.fc3 = nn.Linear(emb_dim * 2, hidden)
        self.fc4 = nn.Linear(hidden, emb_dim)
        self.fc5 = nn.Linear(emb_dim * 2, hidden)
        self.fc6 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout)
        # self.reset_parameters()

    def reset_parameters(self):
        c_stdv = 2. / math.sqrt(self.c_dim)
        x_stdv = 2. / math.sqrt(self.x_dim)
        t_stdv = 2. / math.sqrt(self.t_dim)
        k_f_stdv = 2. / math.sqrt(self.k_f_dim)
        fc_stdv = 2. / math.sqrt(self.emb_dim * 2)
        fc_h_stdv = 2. / math.sqrt(self.hidden)
        self.c_embedding.weight.data.uniform_(-c_stdv, c_stdv)
        self.c_embedding.bias.data.uniform_(-c_stdv, c_stdv)
        self.x_embedding.weight.data.uniform_(-x_stdv, x_stdv)
        self.x_embedding.bias.data.uniform_(-x_stdv, x_stdv)
        self.t_embedding.weight.data.uniform_(-t_stdv, t_stdv)
        self.t_embedding.bias.data.uniform_(-t_stdv, t_stdv)
        self.k_f_embedding.weight.data.uniform_(-k_f_stdv, k_f_stdv)
        self.k_f_embedding.bias.data.uniform_(-k_f_stdv, k_f_stdv)
        self.fc1.weight.data.uniform_(-fc_stdv, fc_stdv)
        self.fc1.bias.data.uniform_(-fc_stdv, fc_stdv)
        self.fc2.weight.data.uniform_(-fc_h_stdv, fc_h_stdv)
        self.fc2.bias.data.uniform_(-fc_h_stdv, fc_h_stdv)
        self.fc3.weight.data.uniform_(-fc_stdv, fc_stdv)
        self.fc3.bias.data.uniform_(-fc_stdv, fc_stdv)
        self.fc4.weight.data.uniform_(-fc_h_stdv, fc_h_stdv)
        self.fc4.bias.data.uniform_(-fc_h_stdv, fc_h_stdv)
        self.fc5.weight.data.uniform_(-fc_stdv, fc_stdv)
        self.fc5.bias.data.uniform_(-fc_stdv, fc_stdv)
        self.fc6.weight.data.uniform_(-fc_h_stdv, fc_h_stdv)
        self.fc6.bias.data.uniform_(-fc_h_stdv, fc_h_stdv)

    def forward(self, c, x, t, k_f, e_cv, e_vc, e_v_veh):
        # TODO:
        #  4. l1 loss with 0. L1=-|score-0| - Done
        #  1. eval: positive
        #  2. disjoint positive
        #  3. positive -> negative
        # initial embedding
        c = self.c_embedding(c)
        v = torch.cat((self.x_embedding(x), self.t_embedding(t)), dim=0)
        k_f = self.k_f_embedding(k_f)

        # v_to_c convolution
        c = torch.cat((c, torch.sparse.mm(e_cv, v)), dim=1)
        # c = torch.sparse.mm(e_cv, v) + c
        c = self.fc2(self.relu(self.fc1(c)))
        # c = self.dropout(c)

        # c_to_v convolution
        v = torch.cat((v, torch.sparse.mm(e_vc, c)), dim=1)
        # v = torch.sparse.mm(e_vc, c) + v
        v = self.fc4(self.relu(self.fc3(v)))

        # print(v)

        # split v by (k, f)
        # veh = {i: torch.zeros(64).unsqueeze(0) for i in veh_x_map.keys()}
        # for key, value in veh_x_map.items():
        #     for row in value:
        #         veh[key] = torch.cat((veh[key], v[row].unsqueeze(0)), dim=0)

        # aggregate to (k, v)
        x = torch.cat((torch.sparse.mm(e_v_veh, v), k_f), dim=1)
        # x = torch.sparse.mm(e_v_veh, v) + k_f
        # print(torch.sparse.mm(e_v_veh, v))
        # print(x)

        x = self.relu(self.fc5(x))
        x = self.fc6(x)

        return x
