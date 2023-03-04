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

    def forward(self, c, x, t, k_f, e_cv, e_vc, e_v_veh):
        # initial embedding
        c = self.c_embedding(c)
        v = torch.cat((self.x_embedding(x), self.t_embedding(t)), dim=0)
        k_f = self.k_f_embedding(k_f)

        # v_to_c convolution
        c = torch.cat((c, torch.sparse.mm(e_cv, v)), dim=1)
        # c = torch.sparse.mm(e_cv, v) + c
        c = self.fc2(self.relu(self.fc1(c)))

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

        x = self.relu(self.fc5(x))
        x = self.fc6(x)

        return x
