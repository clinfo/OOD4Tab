# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, roc_auc_score
from scipy import stats


def MakeRejectionDF(df,pred_proba_name,label_name,score_name,split_num):
    """
    for Make metric-Rejection curve
    Args:
        df (pandas.DataFrame): including model prediction probability, true label(binary), OOD score with each test sample 
        pred_proba_name (str): the name of prediction probability column in df
        label_name (str): the name of binary true label column in df 
        score_name (str): the name of OOD score column in df 
        split_num (int): the number of split 

    Returns:
        pandas.DataFrame: the AUROC and PRAUC in id sample results for each rejection rate [%]
    """
    
    # OOD score min, max
    min_score = df[score_name].min()
    max_score = df[score_name].max()
    
    # thresh
    between_thresh = (max_score - min_score)/split_num
    
    
    num_list = []
    
    for i in range(split_num):
        if i == 0:
            new_thresh = min_score
        else:
            new_thresh = new_thresh + between_thresh
        
        df_id = df[df[score_name] <= new_thresh]
        df_ood = df[df[score_name] > new_thresh]
        
        num_id_sample = df_id.shape[0]
        num_ood_sample = df_ood.shape[0]
        rejection_rate = np.round(100*num_ood_sample/(num_id_sample + num_ood_sample),decimals=1)
        
        # for prediction metric
        df_id_pred_proba = df_id[pred_proba_name]
        df_id_y = df_id[label_name]
        
        if df_id_y.value_counts().shape[0] > 1:
            fpr_,tpr_,_, = roc_curve(df_id_y.tolist(),df_id_pred_proba.tolist())
            auroc_id = auc(fpr_,tpr_)
            precision_,recall_,_, = precision_recall_curve(df_id_y.tolist(),df_id_pred_proba.tolist())
            prauc_id = auc(recall_,precision_)
        else:
            auroc_id = np.nan
            prauc_id = np.nan
        
        num_list.append([new_thresh,num_id_sample,num_ood_sample,rejection_rate,auroc_id,prauc_id])
        
    
    result_df = pd.DataFrame(np.array(num_list),columns=["thresh","num_id","num_ood","rejection_rate","AUROC_id","PRAUC_id"])
    result_df.drop_duplicates(subset = ["num_id","num_ood"],keep="first",inplace=True)
    
    return result_df


def plot_Rejection_curve(rejection_arr, metric_arr, mask, label):
    """
    for plotting Rejection curve
    Args:
        rejection_arr (numpy.array): rejection rate numpy.array
        metric_arr (numpy.array): metric numpy.array 
        mask (int): if rejection rate [%] > mask [%], then not plotting Rejection curve 
        label (str): for plot legend
    """
    rejection_arr_ = np.ma.masked_where(rejection_arr > mask, rejection_arr)
    metric_arr_ = np.ma.masked_where(rejection_arr > mask, metric_arr)
    
    plt.plot(rejection_arr_,metric_arr_,label=label)


def rank_correlation(rejection_arr, metric_arr, nan_policy, method, mask):
    """
    for calculating rank correlation (Spearman or Kendall tau)
    [Ref]
    scipy.stats.spearmanr ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    scipy.stats.kendalltau ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    
    Args:
        rejection_arr (numpy.array): 1D rejection array
        metric_arr (numpy.array): 1D metric result array corresponding to rejection_arr
        nan_policy (str): {'propagate','raise','omit'}
            > 'propagate' returns nan
            > 'raise' throws an error
            > 'omit' performs the calculations ignoring nan values
        method (str): rank correlation method. "spearman" or "kendall"
        mask (int): if rejection rate [%] > mask [%], then excluding from the calculation of rank correlation 

    Returns:
        res: SignificanceResult
         > An object containing attributes:
         >> statistic : Spearman or Kendall correlation coefficient (float)
         >> pvalue : The p-value for a hypothesis test whose null hypothesis is that two samples have no ordinal correlation (float) 
    """
    rejection_arr_ = rejection_arr[rejection_arr <= mask]
    metric_arr_ = metric_arr[rejection_arr <= mask]
    
    if method == "spearman":
        res = stats.spearmanr(rejection_arr_, metric_arr_, nan_policy = nan_policy)
    elif method == "kendall":
        res = stats.kendalltau(rejection_arr_, metric_arr_, nan_policy = nan_policy)
    else:
        NotImplementedError
    
    return res