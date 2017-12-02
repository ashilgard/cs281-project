import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import minimize

primes = [2,3,5,7,11,13]

def create_user_task_ids(df, user_col, task_col, values_col, prime=False, numeric=False):
    y = pd.Series(range(len(df[task_col].unique())), index=df[task_col].unique())
    x = pd.Series(range(len(df[user_col].unique())), index=df[user_col].unique())
    if numeric==False:
        if prime==True:
            v = pd.Series(primes[:len(df[values_col].unique())], index=df[values_col].unique())
        else:
            v = pd.Series(range(len(df[values_col].unique())), index=sorted(df[values_col].unique()))
        df['bin'] = df[values_col].map(v)
    else:
        df['bin'] = df[values_col]
    df['task_id'] = df[task_col].map(y)
    df['uid'] = df[user_col].map(x)
    return df

def compute_individual_dist(df, bin_levels=False, full_dist=True):
    
    completed = np.zeros((len(df['uid'].unique()), len(df['task_id'].unique())))
    completed[df['uid'].values, df['task_id'].values] = 1
    
    values = np.zeros((len(df['uid'].unique()), len(df['task_id'].unique())))
    if bin_levels==True:
        values[df['uid'].values, df['task_id'].values] = df['bin_levels'].values
    else:
        values[df['uid'].values, df['task_id'].values] = df['bin'].values
    
    if bin_levels==True:
        cols_ = primes[:len(df['bin_levels'].unique())]
    else:
        cols_ = primes[:len(df['bin'].unique())]
    #add a unique complex number to each row
    weight = 1j*np.linspace(0, values.shape[1], values.shape[0], endpoint=False)
    individual = values + weight[:, np.newaxis]
    u, ind, cnt = np.unique(individual, return_index=True, return_counts=True)
    #test = (dict(zip(zip(ind//values.shape[1],np.real(u)), cnt)))
    individual = np.zeros_like(values)
    np.put(individual, ((ind - ind%values.shape[1]) + np.real(u)).astype(int), cnt)
    individual = individual/individual[:,1:].sum(axis=1)[:,None]
    individual = individual[:,cols_]
    if full_dist==True:
        full_dist = ~np.any(individual==0, axis=1)
        user_ids = np.where(full_dist==True)
        values = values[full_dist,:]
        completed = completed[full_dist, :]
        individual = individual[full_dist,:]
    
    return completed, values, individual

def compute_deltas(user_index, completed, values, individual, mask, score=False, threshold=0):
    overlapping_task_values = np.multiply(completed[user_index], values)
    overlapping_task_values = overlapping_task_values*mask
    a = np.multiply(overlapping_task_values[user_index], overlapping_task_values)
    #add a unique complex number to each row
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    joint = a + weight[:, np.newaxis]
    u, ind, cnt = np.unique(joint, return_index=True, return_counts=True)

    # test = (dict(zip(zip(ind//a.shape[1],np.real(u)), cnt)))
    joint = np.zeros_like(a)
    np.put(joint, ((ind - ind%a.shape[1]) + np.real(u)).astype(int), cnt)
    datapts = joint[:,1:].sum(axis=1)
    #print(datapts)
    joint = joint + 1
    #joint = joint/joint[:,1:].sum(axis=(1,2))[:,None,None]
                                
    
    num_vals = individual.shape[1]
    prime_vals = np.array(primes[:num_vals])
    idx = prime_vals.reshape(num_vals,1).dot(prime_vals.reshape(1,num_vals)).flatten()
    
    joint = joint[:,idx].reshape(a.shape[0],num_vals,num_vals)
    joint = joint/np.sum(joint, axis=(1,2))[:,None,None]
    dots = np.einsum('...ij,...kl->kil',individual[user_index].reshape(num_vals,1), \
                     individual.reshape(individual.shape[0],num_vals))
    delta_matrices = joint - dots
#     true_matrix_idx = np.logical_and(~np.any(joint==0, axis=(1,2)),~np.any(np.isnan(joint), axis=(1,2)))
    #true_matrix_idx = np.logical_and(np.sum(joint==0, axis=(1,2))<=zero_vals,~np.any(np.isnan(delta_matrices), axis=(1,2)))
#     true_matrix_idx = ~np.any(np.isnan(joint), axis=(1,2))
    true_matrix_idx = datapts>=threshold
    true_matrix_idx[user_index] = False
    #print(true_matrix_idx)
    #print(np.sum(true_matrix_idx))
    clustering_img = np.empty((joint.shape))
    clustering_img.fill(np.nan)
    clustering_img[true_matrix_idx] = delta_matrices[true_matrix_idx]
    if score==False:
        return delta_matrices, true_matrix_idx, clustering_img
    else:
        score_matrices = np.sign(delta_matrices)
        score_matrices[score_matrices==-1]=0
        return score_matrices, true_matrix_idx
    
def expected_score(score_mat, deltas, F, G):
    sigs = F.shape[0]
    pts = deltas * score_mat[np.repeat(F,sigs),np.tile(G,(1,sigs))].reshape(sigs,sigs)
    return np.sum(pts)

def best_expected_score(score_mat, delta_mat):
    F_strat = np.empty(score_mat.shape[0])
    G_strat = np.empty(score_mat.shape[0])
    F_strategies = product(range(score_mat.shape[0]),repeat=score_mat.shape[0])
    max_val = 0
    for F in F_strategies:
        G_strategies = product(range(score_mat.shape[0]),repeat=score_mat.shape[0])
        for G in G_strategies:
            score = expected_score(score_mat, delta_mat, np.array(list(F)),np.array(list(G)))
            if score>=max_val:
#                 print(F,G,score)
                max_val=score
                F_strat = F
                G_strat = G
    return max_val, F_strat, G_strat

def best_expected_score_sp(score_matrices, delta_matrices, tmi):
    
    sigs = score_matrices[0].shape[0]
    
    true_matrix_idx = np.where(tmi==True)[0]
    #print(true_matrix_idx)

    fun = lambda FG: - np.average([np.sum(np.sum(np.einsum('ij,lk->iljk', FG[:sigs**2].reshape(sigs,sigs),FG[(sigs**2)*i:(sigs**2)*(i+1)].reshape(sigs,sigs))*score_matrices[true_matrix_idx[i]],\
                                     axis=(2,3))*delta_matrices[true_matrix_idx[i]]) for i in range(len(true_matrix_idx))])

    cons = ({'type': 'eq', 'fun': lambda FG: np.ones(sigs) - np.sum(FG[:(sigs**2)].reshape(sigs,sigs), axis=1 )}, \
    {'type': 'eq', 'fun': lambda FG: np.sum([np.ones(sigs) - np.sum(FG[(sigs**2)*i:(sigs**2)*(i+1)].reshape(sigs,sigs), axis=1 ) for i in range(len(true_matrix_idx))])})

    bnds = ([(0, 1)]*((sigs**2)+((sigs**2)*len(true_matrix_idx))))
    #s1 = time.time()
    res = minimize(fun, (np.random.uniform(0,1,(sigs**2)+((sigs**2)*len(true_matrix_idx)))), method='SLSQP', bounds=bnds, constraints=cons)
    #print("time:", time.time()-s1)
    return res

def regret(score_matrices, delta_matrices, true_matrix_idx):
    sigs = score_matrices[0].shape[0]
    res = best_expected_score_sp(score_matrices, delta_matrices, true_matrix_idx)
    best_score = -res['fun']
    truth_score = np.average([expected_score(score_matrices[i], delta_matrices[i], np.array(range(sigs)),np.array(range(sigs))) for i in np.where(true_matrix_idx==True)[0]])
    return best_score - truth_score, truth_score