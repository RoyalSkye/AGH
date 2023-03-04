#!/usr/bin/env python

import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import normalize
from utils import load_datasets, sparse_mx_to_torch_sparse_tensor, str2tuple


class CVPRTWDataset(Dataset):
    """
    dataset for cvrptw.
    """
    def __init__(self, iteration=30, dir_path="./data", return_all=False, file_count=0):
        super(CVPRTWDataset, self).__init__()

        # Dataset:
        # file_map; unshared_data["0"]["x"]/["t"]/["label"]/["k_f"]
        # shared_data["0"]["c"]/["e"]/["obj"]/["id_map"]/["veh_map"];
        self.iter = iteration
        self.return_all = return_all
        self.file_map, self.unshared_data, self.shared_data = load_datasets(dir_path=dir_path, file_count=file_count)
        assert len(self.file_map) == len(self.unshared_data) == len(self.shared_data)
        assert len(self.unshared_data["0"]["x"]) == len(self.unshared_data["0"]["t"]) == len(self.unshared_data["0"]["label"]) == len(self.unshared_data["0"]["k_f"])
        self.dataset_size = len(self.unshared_data) * len(self.unshared_data["0"]["label"])

    def __getitem__(self, i):
        # Return:
        # need to re-construct adjacency matrix between c & var.
        if i >= self.dataset_size:
            print("[!] {} index out of range {}-{}".format(self.__class__.__name__, 0, self.dataset_size-1))
            exit(0)
        instance_id = i // self.iter
        iter_id = i % self.iter

        # no (accepted) solution for that iteration
        # print(self.unshared_data[str(instance_id)]["accepted"])
        if not self.unshared_data[str(instance_id)]["accepted"][iter_id] and not self.return_all:
            # print("[!] Warning: No accepted sol for that iter.")
            return None

        c = torch.FloatTensor(self.shared_data[str(instance_id)]["c"])

        try:
            x = torch.FloatTensor(self.unshared_data[str(instance_id)]["x"][iter_id])
            t = torch.FloatTensor(self.unshared_data[str(instance_id)]["t"][iter_id])
            k_f = torch.FloatTensor(self.unshared_data[str(instance_id)]["k_f"][iter_id])
        except Exception as e:
            print(e)
            print(i, instance_id, iter_id)
            print(len(self.unshared_data[str(instance_id)]["x"]), len(self.unshared_data[str(instance_id)]["t"]))

        label = torch.FloatTensor(self.unshared_data[str(instance_id)]["label"][iter_id])
        obj = self.shared_data[str(instance_id)]["obj"]

        # generate sparse adjacency matrix between c and var
        indices = torch.LongTensor([k[:-1] for k in self.shared_data[str(instance_id)]["e"]])
        value = torch.FloatTensor([k[-1:] for k in self.shared_data[str(instance_id)]["e"]]).squeeze(1)
        e_cv = sp.coo_matrix((value, (indices[:, 0], indices[:, 1])), shape=(c.size(0), x.size(0)+t.size(0)), dtype=np.float32)
        # e_cv = torch.sparse_coo_tensor(indices.t(), value, torch.Size([c.size(0), x.size(0)+t.size(0)]))
        e_vc = normalize(e_cv, norm='l1', axis=0)  # normalized by column
        e_cv = normalize(e_cv, norm='l1', axis=1)  # normalized by row
        e_cv = sparse_mx_to_torch_sparse_tensor(e_cv)
        e_vc = sparse_mx_to_torch_sparse_tensor(e_vc)
        e_vc = e_vc.t()

        # construct map
        # load id_map, veh_map
        # veh_x_map: veh_id -> x_id
        id_map = self.shared_data[str(instance_id)]["id_map"]
        id_map = {str2tuple(k): id_map[k] for k in id_map}
        veh_map = self.shared_data[str(instance_id)]["veh_map"]
        veh_map = {str2tuple(k): veh_map[k] for k in veh_map}
        inverse_veh_map = {value: key for key, value in veh_map.items()}
        veh_x_map = {k: [] for k in inverse_veh_map.keys()}
        for key, value in id_map.items():
            if len(key) != 4:
                continue
            # veh_x_map[key[-2], key[-1]].append(value)
            veh_x_map[veh_map[key[-2], key[-1]]].append(value)

        # construct e_v_veh: add x to (k, f), and map it to label
        # print(inverse_veh_map)
        # print(veh_x_map)
        indices = []  # only record for x, not for t
        for row, columns in veh_x_map.items():
            for column in columns:
                indices.append([row, column])
        indices = torch.LongTensor(indices)
        value = torch.ones(len(indices))
        e_v_veh = sp.coo_matrix((value, (indices[:, 0], indices[:, 1])), shape=(len(veh_map), x.size(0)+t.size(0)), dtype=np.float32)
        e_v_veh = normalize(e_v_veh, norm='l1', axis=1)  # normalized by row
        e_v_veh = sparse_mx_to_torch_sparse_tensor(e_v_veh)

        return (c, x, t, k_f, e_cv, e_vc, e_v_veh, label, obj)

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    dataset = CVPRTWDataset(iteration=10, dir_path="./data")
    c, x, t, k_f, e_cv, e_vc, e_v_veh, label, obj = dataset.__getitem__(0)
    # print(c.size())
    # print(e.size())
    # print(label.size())
    # print(len(A))
    # print(e.size())
    # max_c, max_v = 0, 0
    # for i in A:
    #     max_c = max(max_c, i[0])
    #     max_v = max(max_v, i[-1])
    # print(max_c, max_v)
    # exit(0)
