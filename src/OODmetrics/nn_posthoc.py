# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

from sklearn.covariance import EmpiricalCovariance

# for GEM
def sample_estimator(train_X,train_y,model):
    """
    calc for sample mean, precision matrix (the inverse of covariance matrix)
    mahalanobis distance in h space is calculated by them
    Args:
        train_X (torch.Tensor): training samples X, shape is following:[num of samples, num of input features]
        train_y (torch.Tensor): training samples y label, shape is following:[num of samples]
        model : pytorch custom binary classification NN model with logit output

    Returns:
        mu_0, mu_1, precision: the mean of h output in label 0 and 1 (each shape is [1,2]), 
                                precision matrix (the inverse of covariance matrix, shape is [2,2])
    """
    # only for binary classification
    y_label_unique = torch.unique(train_y,sorted=True)
    num_class = y_label_unique.size()[0]
    assert num_class == 2
    # negative label, positive label
    neg_y_label = y_label_unique[0].item()
    pos_y_label = y_label_unique[1].item()
    
    empcov = EmpiricalCovariance(assume_centered=False)
    
    # training samples in each label
    train_X_label_1 = train_X[train_y == pos_y_label]
    train_X_label_0 = train_X[train_y == neg_y_label]
    
    # the num of samples in each label 
    # num_train_X_label_0 = train_X_label_0.shape[0]
    # num_train_X_label_1 = train_X_label_1.shape[0]
    
    model.eval()
    with torch.no_grad():
        
        tmp_logit_0 = model(train_X_label_0)
        logit_for_label_0 = torch.cat((tmp_logit_0,-tmp_logit_0),1)
        mu_0 = logit_for_label_0.mean(dim=0)
        
        # tmp_cov_0 = torch.mm((logit_for_label_0 - mu_0).T, (logit_for_label_0 - mu_0))
        
        tmp_logit_1 = model(train_X_label_1)
        logit_for_label_1 = torch.cat((tmp_logit_1,-tmp_logit_1),1)
        mu_1 = logit_for_label_1.mean(dim=0)
        
        # tmp_cov_1 = torch.mm((logit_for_label_1 - mu_1).T, (logit_for_label_1 - mu_1))
        
        tmp_X = torch.cat(((logit_for_label_1 - mu_1),(logit_for_label_0 - mu_0)),0)
        
        empcov.fit(tmp_X)
        precision = torch.Tensor(empcov.precision_).double()
        

            
    return mu_0,mu_1,precision

def GEM_score(model,mu_0,mu_1,precision_matrix,test_X,weight_for_pos_label=0.5):
    """
    calc for GEM score (Gaussian mixture based energy measurement)
    
    Title: "Provable Guarantees for Understanding Out-of-distribution Detection"
    Link: https://arxiv.org/abs/2112.00787, AAAI 2022
    Args:
        model : pytorch custom binary classification NN model with logit output
        mu_0 (torch.Tensor): sample mean of logit output in label 0, shape is [1,2]
        mu_1 (torch.Tensor): sample mean of logit output in label 1, shape is [1,2]
        precision_matrix (torch.Tensor): the inverse of covariance matrix in sample h space, shape is [2,2] 
        test_X (torch.Tensor): test samples X, shape is following:[num of samples, num of input features]
        weight_for_pos_label (float): 0 ~ 1, weight for positive label

    Returns:
        gem_score(torch.Tensor): GEM OOD score, shape is [num of samples] 
    """
    
    assert 0 < weight_for_pos_label < 1
    w_0 = 1 - weight_for_pos_label
    w_1 = weight_for_pos_label
    
    model.eval()
    with torch.no_grad():
        tmp_test_logit = model(test_X)
        test_logit = torch.cat((tmp_test_logit, -tmp_test_logit),1)
        f_score_0 = -0.5 * torch.matmul(torch.matmul((test_logit-mu_0),precision_matrix),(test_logit-mu_0).T)
        f_score_1 = -0.5 * torch.matmul(torch.matmul((test_logit-mu_1),precision_matrix),(test_logit-mu_1).T)
        
        gem_score = torch.logsumexp(torch.cat((w_0 * f_score_0, w_1 * f_score_1),-1),dim=1)
        
    return np.array(gem_score)


# for Energy-based OOD
def make_negpos_logit_array(positive_logit_list):
    """
    positive_logit_list: before softmax output list
    """
    output_array = np.vstack(
        [-np.array(positive_logit_list), np.array(positive_logit_list)]
    ).T
    return output_array

def energy_ood_score(model,test_X,t=1):
    """
    Energy-based OOD detection
    https://arxiv.org/abs/2010.03759
    """
    model.eval()
    with torch.no_grad():
        tmp_test_logit = model(test_X)
        test_logit = torch.cat((tmp_test_logit, -tmp_test_logit),1)
    
    energy_ood_score = t * torch.logsumexp(test_logit, dim=1)
    
    return np.array(energy_ood_score)