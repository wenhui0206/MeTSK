import numpy as np
import os
import pdb

def flatten_conn_features(conn_pte):
    n_rois = conn_pte.shape[0]
    ind = np.tril_indices(n_rois, k=1)
    return conn_pte[ind[0], ind[1], :].T

def get_features(data):
    conn_mat = np.zeros((data.shape[1], data.shape[1], data.shape[0]))
    for i in range(data.shape[0]):
        conn = np.corrcoef(data[i])
        conn[~np.isfinite(conn)] = 0
        conn_mat[:, :, i] = conn
    return flatten_conn_features(conn_mat)

def load_AD_data(fp, ad_cls=True):
    data = np.load(fp)
    X = get_features(data['features'])
    y = data['labels']
    cdrs = data['cdr']
    if ad_cls:
        mask = np.logical_or(y == 0, y == 1)
        X, y, cdrs = X[mask], y[mask], cdrs[mask]
        mask = cdrs != 0.5
        X, y = X[mask], y[mask]
        n = min(np.sum(y == 0), np.sum(y == 1))
        return np.concatenate((X[y == 0][:n], X[y == 1][:n])), np.concatenate((y[y == 0][:n], y[y == 1][:n]))
    else:
        mask = np.logical_or(y == 0, y == 2)
        X, y = X[mask], y[mask]
        n = min(np.sum(y == 0), np.sum(y == 2))
        return np.concatenate((X[y == 0][:n+10], X[y == 2][:n])), np.concatenate((y[y == 0][:n+10], y[y == 2][:n]))

def call_input(dataname="pte", source="connectivity"):
    base_path = '../../../'
    file_path = os.path.join(base_path, f"{source}_{dataname}.npz") if source != "brainlm" else os.path.join(base_path, f"{source}_{dataname}.npy")

    if source == "connectivity":
        if dataname == "pte":
            epi = flatten_conn_features(np.transpose(np.load(os.path.join(base_path, f"connectivity_pte.npz"))['conn_mat'], (1, 2, 0)))
            nonepi = flatten_conn_features(np.transpose(np.load(os.path.join(base_path, f"connectivity_nonepi.npz"))['conn_mat'], (1, 2, 0)))
            X = np.vstack((nonepi, epi))
            y = np.hstack((np.zeros(36), np.ones(36)))
        elif dataname == "ad":
            return load_AD_data(file_path, ad_cls=True)
        elif dataname == "pd-taowu":
            data = np.load(file_path)
            X = get_features(data['features'])
            y = np.hstack((np.zeros(20), np.ones(20)))
        elif dataname == "pd-neurocon":
            data = np.load(file_path)
            X = get_features(data['features'])
            y = np.hstack((np.zeros(16), np.ones(26)))
        else:
            raise ValueError("Unsupported dataname for connectivity")

    elif source in ["metsk", "ssl"]:
        data = np.load(file_path)['graph_model']
        axis = 1
        X = np.mean(data, axis=axis).reshape((data.shape[0], -1))
        if dataname == "pte":
            y = np.hstack((np.zeros(36), np.ones(36)))
        elif dataname == "pd-taowu":
            y = np.hstack((np.zeros(20), np.ones(20)))
        elif dataname == "pd-neurocon":
            y = np.hstack((np.zeros(16), np.ones(26)))
        elif dataname == "ad":
            y = np.load(os.path.join(base_path, f"connectivity_ad.npz"))['labels']
        else:
            raise ValueError("Unsupported dataname for metsk/ssl")

    elif source == "brainlm":
        X = np.load(file_path)
        if dataname == "pte":
            y = np.hstack((np.zeros(36), np.ones(36)))
        elif dataname == "pd-taowu":
            y = np.hstack((np.zeros(20), np.ones(20)))
        elif dataname == "pd-neurocon":
            y = np.hstack((np.zeros(16), np.ones(26)))
        elif dataname == "ad":
            y = np.load(os.path.join(base_path, f"connectivity_ad.npz"))['labels']
        else:
            raise ValueError("Unsupported dataname for brainlm")

    else:
        raise ValueError("Unsupported source")

    return X, y
