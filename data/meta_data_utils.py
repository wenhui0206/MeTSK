import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from metsk.data.dataset_hcp import DatasetHCPRest
import pdb

def generate_episodes(num_episodes, data, y, num_samp=8, num_cls=2):
    # 8 samples for each domain, each class.
    # 32 episodes in one training batch 32*16*datashape
    epds = []
    epds_y= []
    for i in range(num_episodes):
        perm1 = np.random.permutation(np.where(y[:, 0] == 0)[0])[:num_samp]
        perm2 = np.random.permutation(np.where(y[:, 0] == 1)[0])[:num_samp]

        epds.append(np.concatenate((data[perm1], data[perm2]), axis=0))
        epds_y.append(np.concatenate((y[perm1], y[perm2]), axis=0))

    return np.array(epds), np.array(epds_y)


def calc_populationA(data_all):
    sequence_all=None
    for i in range(data_all.shape[0]):
        z_sequence = data_all[i].squeeze().T
        if sequence_all is None:
            sequence_all = z_sequence
        else:
            sequence_all = np.concatenate((sequence_all, z_sequence), axis=1)
    A = np.corrcoef(sequence_all)
    return A


def get_data(fp1, fp2):
    disease_data = np.load(fp1) #"../data_npz/ADHD_mmp22.npz"
    hc_data = np.load(fp2) # "../data_npz/TDC_mmp22.npz"

    disease_features = np.transpose(disease_data["features"], (0, 2, 1))
    hc_features = np.transpose(hc_data["features"], (0, 2, 1))
    di_shape = disease_features.shape
    hc_shape = hc_features.shape

    data_all = np.concatenate([hc_features.reshape((hc_shape[0],1,hc_shape[1],hc_shape[2],1)), disease_features.reshape((di_shape[0],1,di_shape[1],di_shape[2],1))], axis=0)
    adj_all = np.concatenate([hc_data["conn_mat"], disease_data["conn_mat"]], axis=0)
    label_all = np.concatenate([np.zeros((hc_shape[0], 1)), np.ones((di_shape[0], 1))], axis=0)

    print(data_all.shape, label_all.shape)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for train_ind, test_ind in kfold.split(data_all, label_all):

        return data_all[train_ind], data_all[test_ind], adj_all[train_ind], adj_all[test_ind], label_all[train_ind], label_all[test_ind]


def get_all_data_nosplit(fp1, fp2):
    disease_data = np.load(fp1) #"../data_npz/ADHD_mmp22.npz"
    hc_data = np.load(fp2) # "../data_npz/TDC_mmp22.npz"

    disease_features = np.transpose(disease_data["features"], (0, 2, 1))
    hc_features = np.transpose(hc_data["features"], (0, 2, 1))
    di_shape = disease_features.shape
    hc_shape = hc_features.shape

    data_all = np.concatenate([hc_features.reshape((hc_shape[0],1,hc_shape[1],hc_shape[2],1)), disease_features.reshape((di_shape[0],1,di_shape[1],di_shape[2],1))], axis=0)
    adj_all = np.concatenate([hc_data["conn_mat"], disease_data["conn_mat"]], axis=0)
    label_all = np.concatenate([np.zeros((hc_shape[0], 1)), np.ones((di_shape[0], 1))], axis=0)

    print(data_all.shape, label_all.shape)
    
    return data_all, adj_all, label_all


def get_public_data(filename):
    disease_data = np.load(filename)
    disease_features = np.transpose(disease_data["features"], (0, 2, 1))
    di_shape = disease_features.shape
    data_all = disease_features.reshape((di_shape[0],1,di_shape[1],di_shape[2],1))
    # adj_all = disease_data["conn_mat"]
    label_all = disease_data["labels"].reshape((di_shape[0], 1))

    print(data_all.shape, label_all.shape)

    return data_all, label_all

def get_AD_data(filename, balanced_cls=True):
    disease_data = np.load(filename)
    disease_features = np.transpose(disease_data["features"], (0, 2, 1))
    
    labels = disease_data["labels"]
     # Filter data
    cdrs = disease_data['cdr']
    disease_features = disease_features[cdrs != 0.5]
    labels = labels[cdrs != 0.5]
    disease_features = disease_features[np.logical_or(labels == 0, labels == 1)]
    labels = labels[np.logical_or(labels == 0, labels == 1)]
    
    if balanced_cls:
        # Get equal number of the two labels
        num_samples = min(np.sum(labels == 0), np.sum(labels == 1))
        disease_features = np.concatenate((disease_features[labels == 0][:num_samples], disease_features[labels == 1][:num_samples]))
        labels = np.concatenate((labels[labels == 0][:num_samples], labels[labels == 1][:num_samples]))

    di_shape = disease_features.shape
    data_all = disease_features.reshape((di_shape[0], 1, di_shape[1], di_shape[2], 1))
    # adj_all = disease_data["conn_mat"]
    label_all = labels.reshape((di_shape[0], 1))
    print(data_all.shape, label_all.shape)

    return data_all, label_all


def get_hcp_aal():
    dataset = DatasetHCPRest("/home/wenhui/ImagePTE1/wenhuicu/HCP_stagin/", roi='aal', k_fold=None, smoothing_fwhm=None)
    timeseries = np.array(list(dataset.timeseries_dict.values()))
    data_all = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1, keepdims=True)
  
    di_shape = data_all.shape
    data_all = data_all.reshape((di_shape[0],1,di_shape[1],di_shape[2],1))
    label_all = np.array(dataset.full_label_list).reshape((di_shape[0], 1))
    label_all[label_all=='F'] = 0
    label_all[label_all=='M'] = 1
    label_all = np.array(label_all, dtype=np.int64)
    # num_subjects x input_channels x Len of Time series x num_nodes x num of instances in a frame (=1 here)
    print(data_all.shape, label_all.shape)
    return data_all, label_all

def get_pd_data(filename=None):
    disease_data = np.load(filename)
    disease_features = np.transpose(disease_data["features"], (0, 2, 1))
    di_shape = disease_features.shape
    data_all = disease_features.reshape((di_shape[0],1,di_shape[1],di_shape[2],1))
    # pdb.set_trace()
    # label_all = np.array(dataset.full_label_list).reshape((di_shape[0], 1))
    if di_shape[0] == 42:
        label_all = np.concatenate([np.zeros((16, 1)), np.ones((26, 1))], axis=0)
    else:
        label_all = np.concatenate([np.zeros((20, 1)), np.ones((20, 1))], axis=0)

    # num_subjects x input_channels x Len of Time series x num_nodes x num of instances in a frame (=1 here)
    print(data_all.shape, label_all.shape)
    return data_all, label_all


def get_abide(filename):
    disease_data = np.load(filename)
    disease_features = np.transpose(disease_data["features"], (0, 2, 1))
    di_shape = disease_features.shape
    data_all = disease_features.reshape((di_shape[0],1,di_shape[1],di_shape[2],1))
    # adj_all = disease_data["conn_mat"]
    label_all = disease_data["labels"].reshape((di_shape[0], 1))

    print(data_all.shape, label_all.shape)
    return data_all, label_all
    