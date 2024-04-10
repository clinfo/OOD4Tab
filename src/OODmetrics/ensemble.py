# -*- coding: utf-8 -*-
import numpy as np


def total_uncertainty(pos_neg_pred_array,n_classes = 2):
    """
    Total uncertainty for Ensemble model
    Args:
        pos_neg_pred_array (np.array): an array of predict probability, shape is following:[num of ensembles, test sample, num of class]
        n_classes (int, optional): num of class. Defaults to 2.

    Returns:
        np.array: an array of total uncertainty, shape is following:[test sample,]
    """
    n_trees = pos_neg_pred_array.shape[0] # num of ensemble model
    n_samples = pos_neg_pred_array.shape[1]
    total_unc_list = []
    # c: class loop
    # i: tree loop
    # n: sample loop
    for n in range(n_samples):
        unc = 0
        for c in range(n_classes):
            s = np.sum([pos_neg_pred_array[i, n ,c] for i in range(n_trees)])/n_trees
            if s!=0:
                unc = unc-s*np.log2(s)
        total_unc_list.append(unc)

    return np.array(total_unc_list)

def aleatoric_uncertainty(pos_neg_pred_array, n_classes = 2):
    """
    Aleatoric uncertainty for Ensemble model
    Args:
        pos_neg_pred_array (np.array): an array of predict probability, shape is following:[num of ensembles, test sample, num of class]
        n_classes (int, optional): num of class. Defaults to 2.

    Returns:
        np.array: an array of aleatoric uncertainty, shape is following:[test sample,]
    """
    n_trees = pos_neg_pred_array.shape[0] # num of ensemble model
    n_samples = pos_neg_pred_array.shape[1] # num of test samples
    aleatoric_unc_list = []
    # t: tree loop
    # c: class loop
    # n: sample loop
    for n in range(n_samples):
        unc = 0
        for t in range(n_trees):
            for c in range(n_classes):
                p = pos_neg_pred_array[t, n, c]
                if p!=0:
                    unc = unc + p*np.log2(p)
            
        unc = -unc/n_trees
        aleatoric_unc_list.append(unc)
    return np.array(aleatoric_unc_list)


def epistemic_uncertainty(pos_neg_pred_array, n_classes = 2):
    """
    Epistemic uncertainty for Ensemble model
    Args:
        pos_neg_pred_array (np.array): an array of predict probability, shape is following:[num of ensembles, test sample, num of class]
        n_classes (int, optional): num of class. Defaults to 2.

    Returns:
        np.array: an array of epistemic uncertainty, shape is following:[test sample,]
    """
    _array = pos_neg_pred_array
    _n_class = n_classes
    
    epistemic_uncertainty = total_uncertainty(_array,_n_class) - aleatoric_uncertainty(_array,_n_class)
    
    return epistemic_uncertainty


def make_pred_proba_array(positive_proba_list):
    """
    positive_proba_list: softmax output list
    """
    output_array = np.vstack(
        [1 - np.array(positive_proba_list), np.array(positive_proba_list)]
    ).T
    return output_array