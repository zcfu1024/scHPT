# -*- coding:utf-8 -*-
"""
name:scHPT
function:Test for heterogeneous effects in single-cell perturbation data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
import random
import itertools

from sklearn.metrics import mean_squared_error
from sklearn import cluster as clust
from scipy import sparse as ss
from scipy import optimize as so
from multiprocessing import Pool
import pickle


def plot_ecdf(data_df, tau, title, save=False, save_path=None):
    if save:
        if save_path is None:
            raise Exception("Please provide save_path to save the plot.")
    with plt.rc_context({'figure.figsize': (4, 4), 'font.sans-serif': ['Arial']}):
        shifted_df = data_df[data_df.pert == 1].copy()
        shifted_df['Yobs'] = shifted_df['Yobs'] - tau
        shifted_df['pert'] = 'shifted'
        data_df_drawcdf = pd.concat([data_df.copy(), shifted_df], ignore_index=True, sort=False)
        sns.ecdfplot(data=data_df_drawcdf, x="Yobs", hue="pert")
        plt.title(title)
        if save:
            plt.savefig(os.path.join(save_path, '{}.png'.format(title)), bbox_inches='tight', dpi=1000)
        else:
            plt.show()
        plt.close()


def plot_ecdf_cov(data_df, covs_columns, beta, title):
    with plt.rc_context({'figure.figsize': (4, 4), 'font.sans-serif': ['Arial']}):
        n1 = len(data_df[data_df.pert == 1])
        data1_df = data_df[data_df.pert == 1]
        X_1 = np.concatenate((np.ones(n1).T.reshape(n1, 1), np.array(data1_df[covs_columns].values)), axis=1)
        Yobs_1 = np.array(data1_df['Yobs'].values).reshape(n1, 1)
        shifted_df = data_df[data_df.pert == 1].copy()
        shifted_df['Yobs'] = Yobs_1 - (np.matmul(X_1, beta.reshape(-1, 1)))[:, 0]
        shifted_df['pert'] = 'shifted'
        data_df_drawcdf = pd.concat([data_df.copy(), shifted_df], ignore_index=True, sort=False)
        sns.ecdfplot(data=data_df_drawcdf, x="Yobs", hue="pert")
        plt.title(title)
        plt.show()


def cal_sks(data0, data1, tau):
    data0, data1 = map(np.asarray, (data0, data1))
    n0 = len(data0)
    n1 = len(data1)
    data0 = np.sort(data0)
    data1 = np.sort(data1 - tau)
    data_all = np.concatenate([data0, data1])
    cdf0 = np.searchsorted(data0, data_all, side='right') / (1.0 * n0)
    cdf1 = (np.searchsorted(data1, data_all, side='right')) / (1.0 * n1)
    diff = cdf0 - cdf1
    absd = np.max(np.absolute(diff))
    return absd


def permute_test_df(data_df, n_perm=1000, gamma=0.001, n_search=20, verbose=True, random_state=30):
    n0 = len(data_df[data_df.pert == 0])
    n1 = len(data_df[data_df.pert == 1])
    if n0 == 0 or n1 == 0:
        raise Exception("n_control or n_pert == 0!")
    data0_obs = (data_df[data_df.pert == 0])['Yobs'].values
    data1_obs = (data_df[data_df.pert == 1])['Yobs'].values
    tau_hat = np.mean(data1_obs) - np.mean(data0_obs)
    tau_sd = np.sqrt(np.var(data1_obs, ddof=1) / n1 + np.var(data0_obs, ddof=1) / n0)
    tau_lb, tau_ub = stats.norm.interval(1 - gamma, loc=tau_hat, scale=tau_sd)
    tau_list = list(np.linspace(start=tau_lb, stop=tau_ub, num=n_search))
    tau_list.append(tau_hat)
    pval_list = []
    sks_list = []
    count_num = random_state
    for tau in tau_list:
        sks_obs = cal_sks(data0_obs, data1_obs, tau)
        sks_list.append(sks_obs)
        data_df['Y0'] = data_df['Yobs'].values
        data_df['Y1'] = data_df['Yobs'].values
        # data_df.loc[data_df.pert==1,'Y0']=(data_df[data_df.pert==1])['Yobs'].values-tau
        # data_df.loc[data_df.pert==0,'Y1']=(data_df[data_df.pert==0])['Yobs'].values+tau

        data_df.loc[data_df.pert == 1, 'Y0'] = (data_df.loc[data_df.pert == 1, 'Yobs']).values - tau
        data_df.loc[data_df.pert == 0, 'Y1'] = (data_df.loc[data_df.pert == 0, 'Yobs']).values + tau

        sks_pert_list = []
        for i in range(n_perm):
            count_num += 1
            random.seed(count_num)
            data_df['new_pert'] = random.sample(list(data_df.pert.values), n0 + n1)
            data0 = list(data_df.loc[data_df.new_pert == 0, 'Y0'].values)
            data1 = list(data_df.loc[data_df.new_pert == 1, 'Y1'].values)

            # data0_df=data_df.sample(n=n0,random_state=count_num)
            # data0_df=data_df.sample(n=n0)
            # data0=list(data0_df['Y0'].values)
            # data1=list((data_df.drop(data0_df.index))['Y1'].values)
            sks_pert_list.append(cal_sks(data0, data1, tau))
            if verbose:
                if i % 100 == 0:
                    print('Finish permutation : {}'.format(i))
        # print(sks_pert_list)
        pval = 1 - (np.searchsorted(np.sort(sks_pert_list), sks_obs) / len(sks_pert_list))
        pval_list.append(pval)
    pval_final = np.max(pval_list) + gamma
    if pval_final > 1:
        pval_final = 1
    return pval_final, pval_list, sks_list, tau_list, tau_hat, tau_sd


def cal_min_sks(df, gamma=0.001, n_search=20):
    n0 = len(df[df.pert == 0])
    n1 = len(df[df.pert == 1])
    data0_obs = (df[df.pert == 0])['Yobs'].values
    data1_obs = (df[df.pert == 1])['Yobs'].values
    tau_hat = np.mean(data1_obs) - np.mean(data0_obs)
    tau_sd = np.sqrt(np.var(data1_obs, ddof=1) / n1 + np.var(data0_obs, ddof=1) / n0)
    tau_lb, tau_ub = stats.norm.interval(1 - gamma, loc=tau_hat, scale=tau_sd)
    tau_list = list(np.linspace(start=tau_lb, stop=tau_ub, num=n_search))
    tau_list.append(tau_hat)
    # pval_list=[]
    sks_list = []
    for tau in tau_list:
        sks_obs = cal_sks(data0_obs, data1_obs, tau)
        sks_list.append(sks_obs)
    tau_star = tau_list[np.argmin(sks_list)]
    sks_obs_star = np.min(sks_list)
    return sks_obs_star, tau_star, tau_hat, tau_sd


def permute_test_df2(data_df, n_perm=1000, gamma=0.001, n_search=20, verbose=True, random_state=30):
    n0 = len(data_df[data_df.pert == 0])
    n1 = len(data_df[data_df.pert == 1])
    if n0 == 0 or n1 == 0:
        raise Exception("n_control or n_pert == 0!")
    sks_obs_star, tau_star, tau_hat, tau_sd = cal_min_sks(df=data_df, gamma=gamma, n_search=n_search)
    data_df['Y0'] = data_df['Yobs'].values
    data_df['Y1'] = data_df['Yobs'].values
    data_df.loc[data_df.pert == 1, 'Y0'] = (data_df.loc[data_df.pert == 1, 'Yobs']).values - tau_star
    data_df.loc[data_df.pert == 0, 'Y1'] = (data_df.loc[data_df.pert == 0, 'Yobs']).values + tau_star
    count_num = random_state
    sks_pert_list = []
    for i in range(n_perm):
        count_num += 1
        random.seed(count_num)
        data_df['new_pert'] = random.sample(list(data_df.pert.values), n0 + n1)
        data0 = list(data_df.loc[data_df.new_pert == 0, 'Y0'].values)
        data1 = list(data_df.loc[data_df.new_pert == 1, 'Y1'].values)

        # data0_df=data_df.sample(n=n0,random_state=count_num)
        # data0_df=data_df.sample(n=n0)
        # data0=list(data0_df['Y0'].values)
        # data1=list((data_df.drop(data0_df.index))['Y1'].values)
        sks_pert_list.append(cal_sks(data0, data1, tau_star))
        if verbose:
            if i % 100 == 0:
                print('Finish permutation : {}'.format(i))
    # print(sks_pert_list)
    pval = 1 - (np.searchsorted(np.sort(sks_pert_list), sks_obs_star) / len(sks_pert_list))
    pval_final = pval + gamma
    if pval_final > 1:
        pval_final = 1
    return pval_final, tau_hat, tau_sd


def cal_n_min_sks(df, gamma=0.001, n_search=20, n_min_sks=5):
    n0 = len(df[df.pert == 0])
    n1 = len(df[df.pert == 1])
    data0_obs = (df[df.pert == 0])['Yobs'].values
    data1_obs = (df[df.pert == 1])['Yobs'].values
    tau_hat = np.mean(data1_obs) - np.mean(data0_obs)
    tau_sd = np.sqrt(np.var(data1_obs, ddof=1) / n1 + np.var(data0_obs, ddof=1) / n0)
    tau_lb, tau_ub = stats.norm.interval(1 - gamma, loc=tau_hat, scale=tau_sd)
    tau_list = list(np.linspace(start=tau_lb, stop=tau_ub, num=n_search))
    tau_list.append(tau_hat)
    sks_list = []
    for tau in tau_list:
        sks_obs = cal_sks(data0_obs, data1_obs, tau)
        sks_list.append(sks_obs)
    sks_arr = np.array(sks_list)
    tau_arr = np.array(tau_list)
    sorted_idx = np.argsort(sks_arr)
    top_idx = sorted_idx[:n_min_sks]
    sks_obs_n_min_list = list(sks_arr[top_idx])
    tau_n_min_list = list(tau_arr[top_idx])
    # tau_star = tau_list[np.argmin(sks_list)]
    # sks_obs_star = np.min(sks_list)
    return sks_obs_n_min_list, tau_n_min_list, tau_hat, tau_sd


def permute_test_tau_mp(data_df, tau, sks_obs, n_perm=1000, gamma=0.001, verbose=True, random_state=30):
    data_df['Y0'] = data_df['Yobs'].values
    data_df['Y1'] = data_df['Yobs'].values
    n0 = len(data_df[data_df.pert == 0])
    n1 = len(data_df[data_df.pert == 1])
    data_df.loc[data_df.pert == 1, 'Y0'] = (data_df.loc[data_df.pert == 1, 'Yobs']).values - tau
    data_df.loc[data_df.pert == 0, 'Y1'] = (data_df.loc[data_df.pert == 0, 'Yobs']).values + tau
    count_num = random_state
    sks_pert_list = []
    for i in range(n_perm):
        count_num += 1
        random.seed(count_num)
        data_df['new_pert'] = random.sample(list(data_df.pert.values), n0 + n1)
        data0 = list(data_df.loc[data_df.new_pert == 0, 'Y0'].values)
        data1 = list(data_df.loc[data_df.new_pert == 1, 'Y1'].values)
        sks_pert_list.append(cal_sks(data0, data1, tau))
        if verbose:
            if i % 100 == 0:
                print('Finish permutation : {}'.format(i))
    pval = 1 - (np.searchsorted(np.sort(sks_pert_list), sks_obs) / len(sks_pert_list))
    pval_final = pval + gamma
    if pval_final > 1:
        pval_final = 1
    return pval_final


def run_single_tau(args):
    data_df, tau, sks_obs, n_perm, gamma, verbose, random_state = args
    return permute_test_tau_mp(data_df=data_df.copy(), tau=tau, sks_obs=sks_obs,
                               n_perm=n_perm, gamma=gamma, verbose=verbose, random_state=random_state)


def permute_test_df_mp(data_df, n_perm=1000, gamma=0.001, n_search=20,
                       n_min_sks=5, njobs=5, verbose=True, random_state=30):
    if n_search < n_min_sks:
        raise Exception("n_search must be larger than n_min_sks!")
    n0 = len(data_df[data_df.pert == 0])
    n1 = len(data_df[data_df.pert == 1])
    if n0 == 0 or n1 == 0:
        raise Exception("n_control or n_pert == 0!")
    sks_obs_n_min_list, tau_n_min_list, tau_hat, tau_sd = cal_n_min_sks(df=data_df, gamma=gamma,
                                                                        n_search=n_search, n_min_sks=n_min_sks)
    args_list = [(data_df, tau_n_min_list[i], sks_obs_n_min_list[i],
                  n_perm, gamma, verbose, random_state + i * n_perm) for i in range(n_min_sks)]
    with Pool(processes=njobs) as pool:
        pval_list = pool.map(run_single_tau, args_list)
    pval_final = max(pval_list)
    return pval_final, pval_list, sks_obs_n_min_list, tau_n_min_list, tau_hat, tau_sd


def permute_test_group_df(data_df, n_cell_min=50, n_perm=1000, gamma=0.001, n_search=20, verbose=True, random_state=30):
    cell_count = data_df.groupby(['group', 'pert']).count()
    remove_type = []
    for i in range(len((cell_count[cell_count.Yobs <= n_cell_min].index))):
        remove_type.append((cell_count[cell_count.Yobs <= n_cell_min].index)[i][0])
    remove_type = list(set(remove_type))
    if len(remove_type) > 0:
        print('The number of cells in {} groups is less than threshold {}'.format(len(remove_type), n_cell_min))
        dfs_sks_obs_list = [np.nan] * len(remove_type)
        dfs_tau_hat_list = []
        dfs_tau_sd_list = []
        final_pvals_list = [np.nan] * len(remove_type)
        dfs_tau_star_list = [np.nan] * len(remove_type)
        for i in range(len(remove_type)):
            df_cur_group = data_df[data_df.group == remove_type[i]]
            n0 = len(df_cur_group[df_cur_group.pert == 0])
            n1 = len(df_cur_group[df_cur_group.pert == 1])
            if n0 == 0 or n1 == 0:
                dfs_tau_hat_list.append(np.nan)
                dfs_tau_sd_list.append(np.nan)
            else:
                data0_obs = (df_cur_group[df_cur_group.pert == 0])['Yobs'].values
                data1_obs = (df_cur_group[df_cur_group.pert == 1])['Yobs'].values
                dfs_tau_hat_list.append(np.mean(data1_obs) - np.mean(data0_obs))
                dfs_tau_sd_list.append(np.sqrt(np.var(data1_obs, ddof=1) / n1 + np.var(data0_obs, ddof=1) / n0))
        dfs_sks_obs_arr = np.array(dfs_sks_obs_list)
        dfs_tau_hat_arr = np.array(dfs_tau_hat_list)
        dfs_tau_sd_arr = np.array(dfs_tau_sd_list)
        dfs_tau_star_arr = np.array(dfs_tau_star_list)
        dfs_group_name_arr = np.array(remove_type)
        final_pvals_arr = np.array(final_pvals_list)
        removed_res_df = pd.DataFrame([dfs_group_name_arr, final_pvals_arr, dfs_tau_hat_arr,
                                       dfs_tau_sd_arr, dfs_tau_star_arr, dfs_sks_obs_arr]).T
        removed_res_df.columns = ['group', 'p_val', 'tau_hat', 'tau_sd', 'tau_star', 'sks_obs']
        data_df = data_df[~data_df.group.isin(remove_type)]

    dfs = [y for x, y in data_df.groupby('group')]
    dfs_sks_obs_list = []
    dfs_tau_hat_list = []
    dfs_tau_sd_list = []
    dfs_tau_star_list = []
    dfs_group_name_list = []
    for i in range(len(dfs)):
        df_cur_group = dfs[i].copy()
        dfs_group_name_list.append(df_cur_group['group'].values[0])
        sks_obs_star, tau_star, tau_hat, tau_sd = cal_min_sks(df_cur_group, gamma=gamma, n_search=n_search)

        dfs_tau_hat_list.append(tau_hat)
        dfs_tau_sd_list.append(tau_sd)
        dfs_tau_star_list.append(tau_star)
        dfs_sks_obs_list.append(sks_obs_star)
    # print(dfs_group_name_list)
    group_ranks = np.argsort(dfs_sks_obs_list)[::-1]
    # print(group_ranks)
    # print(dfs_tau_star_list)
    dfs_sks_adjust_arr = np.zeros([n_perm, len(dfs)])
    count_num = random_state
    for b in range(n_perm):
        dfs_sks_perm_list = []
        # dfs_sks_adjust_list=[]
        for j in range(len(dfs)):

            df_cur_group = dfs[j].copy()
            tau_hat_cur_group = dfs_tau_hat_list[j]
            tau_sd_cur_group = dfs_tau_sd_list[j]

            n0 = len(df_cur_group[df_cur_group.pert == 0])
            n1 = len(df_cur_group[df_cur_group.pert == 1])
            count_num += 1
            random.seed(count_num)
            df_cur_group['new_pert'] = random.sample(list(df_cur_group.pert.values), n0 + n1)

            tau_lb_cur_group, tau_ub_cur_group = stats.norm.interval(1 - gamma, loc=tau_hat_cur_group,
                                                                     scale=tau_sd_cur_group)
            tau_list_cur_group = list(np.linspace(start=tau_lb_cur_group, stop=tau_ub_cur_group, num=n_search))
            tau_list_cur_group.append(tau_hat_cur_group)

            sks_list_cur_group = []
            for tau in tau_list_cur_group:
                # tau_star_cur_group=dfs_tau_star_list[j]
                # print(tau_star_cur_group)

                df_cur_group['Y0'] = df_cur_group['Yobs'].values
                df_cur_group['Y1'] = df_cur_group['Yobs'].values
                df_cur_group.loc[df_cur_group.pert == 1, 'Y0'] = (df_cur_group.loc[
                    df_cur_group.pert == 1, 'Yobs']).values - tau
                df_cur_group.loc[df_cur_group.pert == 0, 'Y1'] = (df_cur_group.loc[
                    df_cur_group.pert == 0, 'Yobs']).values + tau

                data0 = list(df_cur_group.loc[df_cur_group.new_pert == 0, 'Y0'].values)
                data1 = list(df_cur_group.loc[df_cur_group.new_pert == 1, 'Y1'].values)
                sks_list_cur_group.append(cal_sks(data0, data1, tau))
            dfs_sks_perm_list.append(np.min(sks_list_cur_group))

        dfs_sks_perm_arr = np.array(dfs_sks_perm_list)
        # print(dfs_sks_perm_arr)
        for k in range(len(dfs_sks_perm_list)):
            dfs_sks_adjust_arr[b, k] = np.max(dfs_sks_perm_arr[group_ranks[k:]])
            # print(dfs_sks_perm_arr[group_ranks[k:]])
        # print(dfs_sks_adjust_arr[b,:])
        if verbose:
            if b % 100 == 0:
                print('Finish permutation : {}'.format(b))
    final_pvals_list = []
    for grp in range(len(dfs)):
        if grp == 0:
            sks_obs = dfs_sks_obs_list[group_ranks[grp]]
            null_dis = np.sort(dfs_sks_adjust_arr[:, grp])
            final_pvals_list.append(1 - (np.searchsorted(null_dis, sks_obs) / n_perm))
        else:
            sks_obs = dfs_sks_obs_list[group_ranks[grp]]
            null_dis = np.sort(dfs_sks_adjust_arr[:, grp])
            final_pval = np.max([1 - (np.searchsorted(null_dis, sks_obs) / n_perm), final_pvals_list[grp - 1]])
            final_pvals_list.append(final_pval)

    dfs_sks_obs_arr = np.array(dfs_sks_obs_list)[group_ranks]
    dfs_tau_hat_arr = np.array(dfs_tau_hat_list)[group_ranks]
    dfs_tau_sd_arr = np.array(dfs_tau_sd_list)[group_ranks]
    dfs_tau_star_arr = np.array(dfs_tau_star_list)[group_ranks]
    dfs_group_name_arr = np.array(dfs_group_name_list)[group_ranks]
    final_pvals_arr = np.array(final_pvals_list)

    final_res_df = pd.DataFrame([dfs_group_name_arr, final_pvals_arr, dfs_tau_hat_arr,
                                 dfs_tau_sd_arr, dfs_tau_star_arr, dfs_sks_obs_arr]).T
    final_res_df.columns = ['group', 'p_val', 'tau_hat', 'tau_sd', 'tau_star', 'sks_obs']
    if len(remove_type) > 0:
        final_res_df = pd.concat([final_res_df, removed_res_df], ignore_index=True)
    return final_res_df


def cal_sks_cov(data0, data1, cov1, beta):
    data0, data1 = map(np.asarray, (data0, data1))
    beta = beta.reshape(-1, 1)
    n0 = len(data0)
    n1 = len(data1)
    data0 = np.sort(data0)
    data1 = np.sort(data1 - (np.matmul(cov1, beta))[:, 0])
    data_all = np.concatenate([data0, data1])
    cdf0 = np.searchsorted(data0, data_all, side='right') / (1.0 * n0)
    cdf1 = (np.searchsorted(data1, data_all, side='right')) / (1.0 * n1)
    diff = cdf0 - cdf1
    absd = np.max(np.absolute(diff))
    return absd


def sample_ellipsoid(mean, cov, n_search, gamma=0.001):
    eigenvals, eigenvecs = np.linalg.eig(cov)
    semi_lens = np.sqrt(stats.chi2.ppf(1 - gamma, len(cov))) * np.sqrt(eigenvals)
    X_dims = len(cov)
    paras_list = []
    for i in range(X_dims):
        if i == 0:
            paras_list.append(list(np.linspace(0, 1, num=n_search[0] + 1)[1:]))
        elif i == X_dims - 1:
            # paras=list(np.linspace(0, 2*np.pi, num=5))
            # paras[-1]=paras[-1]-0.001
            paras_list.append(list(np.linspace(0, 2 * np.pi, num=n_search[i], endpoint=False)))
        else:
            paras_list.append(list(np.linspace(0, np.pi, num=n_search[i])))
    r_phi_points = list(itertools.product(*paras_list))
    cartesian_points = []
    for point in r_phi_points:
        point = list(point)
        cartesian_axes = []
        for i in range(X_dims):
            if i == 0:
                cartesian_axes.append(point[0] * np.cos(point[1]))
            elif i == X_dims - 1:
                cartesian_axes.append(np.prod([point[0]] + [np.sin(term) for term in point[1:]]))
            else:
                cartesian_axes.append(
                    np.prod([point[0]] + [np.sin(term) for term in point[1:(i + 1)]] + [np.cos(point[i + 1])]))
        cartesian_points.append(cartesian_axes)
    cartesian_points = np.array(cartesian_points)
    sampled_points = (np.matmul(eigenvecs, (np.matmul(cartesian_points, np.diag(semi_lens))).T)).T + mean[:, 0]
    unique_points = [list(i) for i in sampled_points]
    unique_points = np.unique(unique_points, axis=0)
    unique_points = np.concatenate((unique_points, mean.T), axis=0)
    return unique_points


def permute_test_cov_df(data_df, covs_columns, n_perm=100, gamma=0.001, n_search=[5, 10]):
    if len(covs_columns) != len(n_search) - 1:
        raise Exception("The length of 'covs_columns' is not match to the length of 'n_search'!")
    n0 = len(data_df[data_df.pert == 0])
    n1 = len(data_df[data_df.pert == 1])
    data0_df = data_df[data_df.pert == 0]
    data1_df = data_df[data_df.pert == 1]
    X_0 = np.concatenate((np.ones(n0).T.reshape(n0, 1), np.array(data0_df[covs_columns].values)), axis=1)
    X_1 = np.concatenate((np.ones(n1).T.reshape(n1, 1), np.array(data1_df[covs_columns].values)), axis=1)
    Yobs_0 = np.array(data0_df['Yobs'].values).reshape(n0, 1)
    Yobs_1 = np.array(data1_df['Yobs'].values).reshape(n1, 1)
    data_0_obs = data0_df['Yobs'].values
    data_1_obs = data1_df['Yobs'].values
    Sxx_0_inv = np.linalg.inv(np.matmul(X_0.T, X_0) / n0)
    Sxx_1_inv = np.linalg.inv(np.matmul(X_1.T, X_1) / n1)
    Sx_0 = np.matmul(X_0.T, Yobs_0) / n0
    Sx_1 = np.matmul(X_1.T, Yobs_1) / n1
    beta_hat = np.matmul(Sxx_1_inv, Sx_1) - np.matmul(Sxx_0_inv, Sx_0)
    e_hat_0 = Yobs_0 - np.matmul(X_0, beta_hat)
    e_hat_1 = Yobs_1 - np.matmul(X_1, beta_hat)
    S_ex_hat_0 = np.cov((e_hat_0 * X_0).T)
    S_ex_hat_1 = np.cov((e_hat_1 * X_1).T)
    cov_beta_hat = np.matmul(np.matmul(Sxx_1_inv, S_ex_hat_1 / n1), Sxx_1_inv) + np.matmul(
        np.matmul(Sxx_0_inv, S_ex_hat_0 / n0), Sxx_0_inv)
    sampled_betas = sample_ellipsoid(mean=beta_hat, cov=cov_beta_hat, n_search=n_search, gamma=gamma)

    pval_list = []
    sks_list = []
    count_num = 1000

    for i in range(len(sampled_betas)):
        beta = sampled_betas[i]
        sks_obs = cal_sks_cov(data_0_obs, data_1_obs, X_1, beta)
        sks_list.append(sks_obs)

        data_df['Y0'] = data_df['Yobs'].values
        data_df['Y1'] = data_df['Yobs'].values

        data_df.loc[data_df.pert == 1, 'Y0'] = (data_df.loc[data_df.pert == 1, 'Yobs']).values - \
                                               (np.matmul(X_1, beta.reshape(-1, 1)))[:, 0]
        data_df.loc[data_df.pert == 0, 'Y1'] = (data_df.loc[data_df.pert == 0, 'Yobs']).values + \
                                               (np.matmul(X_0, beta.reshape(-1, 1)))[:, 0]

        sks_pert_list = []
        for j in range(n_perm):
            count_num += 1
            random.seed(count_num)
            data_df['new_pert'] = random.sample(list(data_df.pert.values), n0 + n1)
            data0 = list(data_df.loc[data_df.new_pert == 0, 'Y0'].values)
            data1 = list(data_df.loc[data_df.new_pert == 1, 'Y1'].values)
            cov1 = np.concatenate((np.ones(len(data1)).T.reshape(len(data1), 1),
                                   np.array((data_df.loc[data_df.new_pert == 1, covs_columns]).values)), axis=1)
            sks_pert_list.append(cal_sks_cov(data0, data1, cov1, beta))
        pval = 1 - (np.searchsorted(np.sort(sks_pert_list), sks_obs) / len(sks_pert_list))
        pval_list.append(pval)

    pval_final = np.max(pval_list) + gamma
    if pval_final > 1:
        pval_final = 1
    return pval_final, pval_list, sks_list, sampled_betas, beta_hat, cov_beta_hat


def permute_test_cov_df2(data_df, covs_columns, n_perm=1000, gamma=0.001, n_search=[5, 10], random_state=30):
    if len(covs_columns) != len(n_search) - 1:
        raise Exception("The length of 'covs_columns' is not match to the length of 'n_search'!")
    n0 = len(data_df[data_df.pert == 0])
    n1 = len(data_df[data_df.pert == 1])
    data0_df = data_df[data_df.pert == 0]
    data1_df = data_df[data_df.pert == 1]
    X_0 = np.concatenate((np.ones(n0).T.reshape(n0, 1), np.array(data0_df[covs_columns].values)), axis=1)
    X_1 = np.concatenate((np.ones(n1).T.reshape(n1, 1), np.array(data1_df[covs_columns].values)), axis=1)
    Yobs_0 = np.array(data0_df['Yobs'].values).reshape(n0, 1)
    Yobs_1 = np.array(data1_df['Yobs'].values).reshape(n1, 1)
    data_0_obs = data0_df['Yobs'].values
    data_1_obs = data1_df['Yobs'].values
    Sxx_0_inv = np.linalg.inv(np.matmul(X_0.T, X_0) / n0)
    Sxx_1_inv = np.linalg.inv(np.matmul(X_1.T, X_1) / n1)
    Sx_0 = np.matmul(X_0.T, Yobs_0) / n0
    Sx_1 = np.matmul(X_1.T, Yobs_1) / n1
    beta_hat = np.matmul(Sxx_1_inv, Sx_1) - np.matmul(Sxx_0_inv, Sx_0)
    e_hat_0 = Yobs_0 - np.matmul(X_0, beta_hat)
    e_hat_1 = Yobs_1 - np.matmul(X_1, beta_hat)
    S_ex_hat_0 = np.cov((e_hat_0 * X_0).T)
    S_ex_hat_1 = np.cov((e_hat_1 * X_1).T)
    cov_beta_hat = np.matmul(np.matmul(Sxx_1_inv, S_ex_hat_1 / n1), Sxx_1_inv) + np.matmul(
        np.matmul(Sxx_0_inv, S_ex_hat_0 / n0), Sxx_0_inv)
    sampled_betas = sample_ellipsoid(mean=beta_hat, cov=cov_beta_hat, n_search=n_search, gamma=gamma)

    # pval_list=[]
    sks_list = []
    count_num = random_state

    for i in range(len(sampled_betas)):
        beta = sampled_betas[i]
        sks_obs = cal_sks_cov(data_0_obs, data_1_obs, X_1, beta)
        sks_list.append(sks_obs)

    beta_star = sampled_betas[np.argmin(sks_list)]
    sks_obs_star = np.min(sks_list)

    data_df['Y0'] = data_df['Yobs'].values
    data_df['Y1'] = data_df['Yobs'].values

    data_df.loc[data_df.pert == 1, 'Y0'] = (data_df.loc[data_df.pert == 1, 'Yobs']).values - \
                                           (np.matmul(X_1, beta_star.reshape(-1, 1)))[:, 0]
    data_df.loc[data_df.pert == 0, 'Y1'] = (data_df.loc[data_df.pert == 0, 'Yobs']).values + \
                                           (np.matmul(X_0, beta_star.reshape(-1, 1)))[:, 0]

    sks_pert_list = []
    for i in range(n_perm):
        count_num += 1
        random.seed(count_num)
        data_df['new_pert'] = random.sample(list(data_df.pert.values), n0 + n1)
        data0 = list(data_df.loc[data_df.new_pert == 0, 'Y0'].values)
        data1 = list(data_df.loc[data_df.new_pert == 1, 'Y1'].values)
        cov1 = np.concatenate((np.ones(len(data1)).T.reshape(len(data1), 1),
                               np.array((data_df.loc[data_df.new_pert == 1, covs_columns]).values)), axis=1)
        sks_pert_list.append(cal_sks_cov(data0, data1, cov1, beta_star))
    pval = 1 - (np.searchsorted(np.sort(sks_pert_list), sks_obs_star) / len(sks_pert_list))
    # pval_list.append(pval)

    pval_final = pval + gamma
    if pval_final > 1:
        pval_final = 1
    return pval_final, sks_list, sampled_betas, beta_hat, cov_beta_hat


def permute_test_cov_beta_mp(data_df, covs_columns, beta, sks_obs, n_perm=1000, gamma=0.001, verbose=True,
                             random_state=30):
    data_df['Y0'] = data_df['Yobs'].values
    data_df['Y1'] = data_df['Yobs'].values
    n0 = len(data_df[data_df.pert == 0])
    n1 = len(data_df[data_df.pert == 1])
    data0_df = data_df[data_df.pert == 0]
    data1_df = data_df[data_df.pert == 1]
    X_0 = np.concatenate((np.ones(n0).T.reshape(n0, 1), np.array(data0_df[covs_columns].values)), axis=1)
    X_1 = np.concatenate((np.ones(n1).T.reshape(n1, 1), np.array(data1_df[covs_columns].values)), axis=1)
    data_df.loc[data_df.pert == 1, 'Y0'] = (data_df.loc[data_df.pert == 1, 'Yobs']).values - \
                                           (np.matmul(X_1, beta.reshape(-1, 1)))[:, 0]
    data_df.loc[data_df.pert == 0, 'Y1'] = (data_df.loc[data_df.pert == 0, 'Yobs']).values + \
                                           (np.matmul(X_0, beta.reshape(-1, 1)))[:, 0]

    count_num = random_state
    sks_pert_list = []
    for i in range(n_perm):
        count_num += 1
        random.seed(count_num)
        data_df['new_pert'] = random.sample(list(data_df.pert.values), n0 + n1)
        data0 = list(data_df.loc[data_df.new_pert == 0, 'Y0'].values)
        data1 = list(data_df.loc[data_df.new_pert == 1, 'Y1'].values)
        cov1 = np.concatenate((np.ones(len(data1)).T.reshape(len(data1), 1),
                               np.array((data_df.loc[data_df.new_pert == 1, covs_columns]).values)), axis=1)
        sks_pert_list.append(cal_sks_cov(data0, data1, cov1, beta))
        if verbose:
            if i % 100 == 0:
                print('Finish permutation : {}'.format(i))
    pval = 1 - (np.searchsorted(np.sort(sks_pert_list), sks_obs) / len(sks_pert_list))
    pval_final = pval + gamma
    if pval_final > 1:
        pval_final = 1
    return pval_final


def run_single_beta(args):
    data_df, covs_columns, beta, sks_obs, n_perm, gamma, verbose, random_state = args
    return permute_test_cov_beta_mp(data_df=data_df.copy(), covs_columns=covs_columns, beta=beta,
                                    sks_obs=sks_obs, n_perm=n_perm, gamma=gamma, verbose=verbose,
                                    random_state=random_state)


def permute_test_cov_df_mp(data_df, covs_columns, n_perm=1000, gamma=0.001, n_search=[5, 10], n_min_sks=10, njobs=10,
                           verbose=True, random_state=30):
    if len(covs_columns) != len(n_search) - 1:
        raise Exception("The length of 'covs_columns' is not match to the length of 'n_search'!")
    n0 = len(data_df[data_df.pert == 0])
    n1 = len(data_df[data_df.pert == 1])
    data0_df = data_df[data_df.pert == 0]
    data1_df = data_df[data_df.pert == 1]
    X_0 = np.concatenate((np.ones(n0).T.reshape(n0, 1), np.array(data0_df[covs_columns].values)), axis=1)
    X_1 = np.concatenate((np.ones(n1).T.reshape(n1, 1), np.array(data1_df[covs_columns].values)), axis=1)
    Yobs_0 = np.array(data0_df['Yobs'].values).reshape(n0, 1)
    Yobs_1 = np.array(data1_df['Yobs'].values).reshape(n1, 1)
    data_0_obs = data0_df['Yobs'].values
    data_1_obs = data1_df['Yobs'].values
    Sxx_0_inv = np.linalg.inv(np.matmul(X_0.T, X_0) / n0)
    Sxx_1_inv = np.linalg.inv(np.matmul(X_1.T, X_1) / n1)
    Sx_0 = np.matmul(X_0.T, Yobs_0) / n0
    Sx_1 = np.matmul(X_1.T, Yobs_1) / n1
    beta_hat = np.matmul(Sxx_1_inv, Sx_1) - np.matmul(Sxx_0_inv, Sx_0)
    e_hat_0 = Yobs_0 - np.matmul(X_0, beta_hat)
    e_hat_1 = Yobs_1 - np.matmul(X_1, beta_hat)
    S_ex_hat_0 = np.cov((e_hat_0 * X_0).T)
    S_ex_hat_1 = np.cov((e_hat_1 * X_1).T)
    cov_beta_hat = np.matmul(np.matmul(Sxx_1_inv, S_ex_hat_1 / n1), Sxx_1_inv) + np.matmul(
        np.matmul(Sxx_0_inv, S_ex_hat_0 / n0), Sxx_0_inv)
    sampled_betas = sample_ellipsoid(mean=beta_hat, cov=cov_beta_hat, n_search=n_search, gamma=gamma)
    sks_list = []
    for i in range(len(sampled_betas)):
        beta = sampled_betas[i]
        sks_obs = cal_sks_cov(data_0_obs, data_1_obs, X_1, beta)
        sks_list.append(sks_obs)

    sks_arr = np.array(sks_list)
    sorted_idx = np.argsort(sks_arr)
    top_idx = sorted_idx[:n_min_sks]
    sks_obs_n_min_list = list(sks_arr[top_idx])
    beta_n_min_list = sampled_betas[top_idx]
    args_list = [(data_df, covs_columns, beta_n_min_list[i], sks_obs_n_min_list[i],
                  n_perm, gamma, verbose, random_state + i * n_perm) for i in range(n_min_sks)]
    with Pool(processes=njobs) as pool:
        pval_list = pool.map(run_single_beta, args_list)
    pval_final = max(pval_list)
    return pval_final, pval_list, sks_obs_n_min_list, sampled_betas, beta_hat, cov_beta_hat


def permute_test(adata, Y_column, T_column, pert_name, ctrl_name, n_perm=1000, gamma=0.001, n_search=20,
                 n_min_sks=5, njobs=5, verbose=True, random_state=30):
    """
    Permutation test for heterogeneous effects (Test1)
    Parameters
    ----------
    adata: Anndata
    single-cell data
    Y_column: string
    The column name of response (Y) in metadata 'adata.obs'
    T_column: string
    The column name of perturbation (T) in metadata 'adata.obs'
    pert_name: string
    The name of perturbed cells
    ctrl_name:string
    The name of control cells
    n_perm: int
    The number of permutations
    gamma: float
    The confidence level. Constructing (1-gamma) confidence interval.
    n_search: int
    The number of points selected in pvalue aggregation.
    Test statistics are calculated on n_search points.
    n_min_sks: int
    The number of points selected in pvalue calculation.
    Pvalues are calculated on top n_min_sks points with smallest SKS.
    njobs: int
    Number of parallel jobs
    verbose: bool
    Print progress or not
    random_state: int
    Random seed
    Returns
    -------
    Test1 pvalue; pvalues, SKS, and corresponding taus in aggregation; estimation of tau and sd(tau)
    """
    adata_subset = adata[adata.obs[T_column].isin([pert_name, ctrl_name])]
    data_df = (pd.DataFrame([list(adata_subset.obs[Y_column]),
                             list(adata_subset.obs[T_column])])).T
    data_df.columns = ['Yobs', 'pert']
    pert_dict = {pert_name: 1, ctrl_name: 0}
    data_df["pert"].replace(pert_dict, inplace=True)
    data_df["Yobs"] = pd.to_numeric(data_df["Yobs"])
    data_df["pert"] = pd.to_numeric(data_df["pert"])
    pval_final, pval_list, sks_list, tau_list, tau_hat, tau_sd = permute_test_df_mp(data_df, n_perm, gamma,
                                                                                    n_search,
                                                                                    n_min_sks,
                                                                                    njobs,
                                                                                    verbose,
                                                                                    random_state)
    return pval_final, pval_list, sks_list, tau_list, tau_hat, tau_sd


def permute_test_group(adata, Y_column, T_column, group_column, pert_name, ctrl_name,
                       n_cell_min=50, n_perm=1000, gamma=0.001, n_search=20,
                       verbose=True, random_state=30):
    """
    Permutation test for subgroup heterogeneous effects (Test2)
    Parameters
    ----------
    adata: Anndata
    single-cell data
    Y_column: string
    The column name of response (Y) in metadata 'adata.obs'
    T_column: string
    The column name of perturbation (T) in metadata 'adata.obs'
    group_column: string
    The column name of subgroup (e.g. cell lines) in metadata 'adata.obs'
    pert_name: string
    The name of perturbed cells
    ctrl_name: string
    The name of control cells
    n_cell_min: int
    The threshold of number cells in each subgroup.
    n_perm: int
    The number of permutations
    gamma: float
    The confidence level. Constructing (1-gamma) confidence interval.
    n_search: int
    The number of points selected in pvalue aggregation.
    Test statistics are calculated on n_search points.
    verbose: bool
    Print progress or not
    random_state: int
    Random seed
    Returns
    -------
    Test2 pvalues;  estimations of tau and sd(tau) in each subgroup
    """
    adata_subset = adata[adata.obs[T_column].isin([pert_name, ctrl_name])]
    data_df = (pd.DataFrame([list(adata_subset.obs[Y_column]),
                             list(adata_subset.obs[T_column]),
                             list(adata_subset.obs[group_column])])).T
    data_df.columns = ['Yobs', 'pert', 'group']
    pert_dict = {pert_name: 1, ctrl_name: 0}
    data_df["pert"].replace(pert_dict, inplace=True)
    data_df["Yobs"] = pd.to_numeric(data_df["Yobs"])
    data_df["pert"] = pd.to_numeric(data_df["pert"])
    final_res_df = permute_test_group_df(data_df, n_cell_min, n_perm, gamma, n_search, verbose, random_state)
    return final_res_df


def permute_test_cov(adata, Y_column, T_column, covs_columns, pert_name, ctrl_name,
                     n_perm=1000, gamma=0.001, n_search=[10, 10], n_min_sks=10, njobs=10,
                     verbose=True, random_state=30):
    """
    Permutation test for continuous covariates-related heterogeneous effects (Test3)
    Parameters
    ----------
    adata: Anndata
    single-cell data
    Y_column: string
    The column name of response (Y) in metadata 'adata.obs'
    T_column: string
    The column name of perturbation (T) in metadata 'adata.obs'
    covs_columns: list
    The columns names of interested covariates in metadata 'adata.obs'
    pert_name: string
    The name of perturbed cells
    ctrl_name:string
    The name of control cells
    n_perm: int
    The number of permutations
    gamma: float
    The confidence level. Constructing (1-gamma) confidence region.
    n_search: list
    The number of points (grid numbers) selected in pvalue aggregation.
    Test statistics are calculated on n_search[0]*n_search[1]*... grid points.
    n_min_sks: int
    The number of points selected in pvalue calculation.
    Pvalues are calculated on top n_min_sks points with smallest SKS.
    njobs: int
    Number of parallel jobs
    verbose: bool
    Print progress or not
    random_state: int
    Random seed
    Returns
    -------
    Test3 pvalue; pvalues, SKS, and betas in aggregation; estimation of beta and cov(beta)
    """
    adata_subset = adata[adata.obs[T_column].isin([pert_name, ctrl_name])]
    data_df = (pd.DataFrame([list(adata_subset.obs[Y_column]),
                             list(adata_subset.obs[T_column])])).T
    data_df.columns = ['Yobs', 'pert']
    pert_dict = {pert_name: 1, ctrl_name: 0}
    data_df["pert"].replace(pert_dict, inplace=True)
    data_df["Yobs"] = pd.to_numeric(data_df["Yobs"])
    data_df["pert"] = pd.to_numeric(data_df["pert"])
    for cov_name in covs_columns:
        data_df[cov_name] = adata_subset.obs[cov_name].values
        data_df[cov_name] = pd.to_numeric(data_df[cov_name])
    pval_final, pval_list, sks_list, sampled_betas, beta_hat, cov_beta_hat = permute_test_cov_df_mp(data_df,
                                                                                                        covs_columns,
                                                                                                        n_perm, gamma,
                                                                                                        n_search,
                                                                                                        n_min_sks,
                                                                                                        njobs,
                                                                                                        verbose,
                                                                                                        random_state)
    return pval_final, pval_list, sks_list, sampled_betas, beta_hat, cov_beta_hat


################# Part1: oNMF
def minibatchkmeans(m, k):
    """
    Perform mini-batch kmeans
    Parameters
    ----------
    m: sparse matrix
    Normalized data
    k: int
    Numbers of clusters
    Returns
    -------
    Cluster centroids, cluster memberships
    """
    model = clust.MiniBatchKMeans(n_clusters=k, random_state=0)  # prepare k means model
    model.fit(m)  # fit with data
    return model.cluster_centers_.T, model.predict(m)


def oNMF(X, k, n_iters=500, verbose=1, residual=1e-4, tof=1e-4):
    """
    Perform non-negative orthogonal matrix factorization
    Revised based on https://github.com/thomsonlab/popalign
    Parameters
    ----------
    X: sparse matrix
    Normalized data
    k: int
    Number of factors
    n_iters: int
    Maximum number of iterations
    verbose: boolean
    Print iteration numbers if 1(True)
    residual: float
    Algorithm converged if the reconstruction error is below the residual values
    tof: float
    Tolerance of the stopping condition
    Returns
    -------
    factor loadings (W) and factor-level expressions (Z)
    """
    r, c = X.shape  # r number of features(genes), c number of samples (cells)
    A, inx = minibatchkmeans(X.T, k)  # Initialize the features (centers of the kmeans clusters)
    orthogonal = [1, 0]  # orthogonality constraints

    Y = ss.csc_matrix((np.ones(c), (inx, range(c))), shape=(k, c)).todense()
    Y = Y + 0.2
    if np.sum(orthogonal) == 2:
        S = A.T.dot(X.dot(Y.T))
    else:
        S = np.eye(k)

    X = X.todense()
    XfitPrevious = np.inf
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(n_iters):
            if orthogonal[0] == 1:
                A = np.multiply(A, (X.dot(Y.T.dot(S.T))) / (A.dot(A.T.dot(X.dot(Y.T.dot(S.T))))))
            else:
                A = np.multiply(A, (X.dot(Y.T)) / (A.dot(Y.dot(Y.T))))
            A = np.nan_to_num(A)
            A = np.maximum(A, np.spacing(1))

            if orthogonal[1] == 1:
                Y = np.multiply(Y, (S.T.dot(A.T.dot(X))) / (S.T.dot(A.T.dot(X.dot(Y.T.dot(Y))))))
            else:
                Y = np.multiply(Y, (A.T.dot(X)) / (A.T.dot(A.dot(Y))))
            Y = np.nan_to_num(Y)
            Y = np.maximum(Y, np.spacing(1))

            if np.sum(orthogonal) == 2:
                S = np.multiply(S, (A.T.dot(X.dot(Y.T))) / (A.T.dot(A.dot(S.dot(Y.dot(Y.T))))))
                S = np.maximum(S, np.spacing(1))

            if np.mod(i, 100) == 0 or i == n_iters - 1:
                if verbose:
                    print('......... Iteration #%d' % i)
                XfitThis = A.dot(S.dot(Y))
                fitRes = np.linalg.norm(XfitPrevious - XfitThis, ord='fro')
                XfitPrevious = XfitThis
                curRes = np.linalg.norm(X - XfitThis, ord='fro')
                if tof >= fitRes or residual >= curRes or i == n_iters - 1:
                    print('Orthogonal NMF performed with %d iterations\n' % (i + 1))
                    break
    return A, Y  # return factor loadings and projection for the cells in X


def nnls(W, V):
    """
    Project 'V' onto 'W' with non-negative least squares
    Parameters
    ----------
    W: array
    Factor loadings
    V: array
    Gene expression vector for a cell
    Returns
    -------
    The factor-level expression of a cell
    """
    return so.nnls(W, V)[0]


def reconstruction_errors(X, loadings, njobs=2):
    """
    Compute the mean square errors between the original data 'X'
    and reconstructed matrices using projection spaces from the list of loadings
    Parameters
    ----------
    X: sparse matrix
    Normalized data
    loadings: list
    List of factor loading arrays to try
    njobs: int
    Number of parallel jobs
    Returns
    -------
    The reconstruction errors, projections (factor-level expression)
    """
    errors = []
    projs = []
    D = X.toarray()  # dense matrix to compute error
    for j in range(len(loadings)):  # For each factor loading matrix j in loadings list
        print('Progress: %d of %d' % ((j + 1), len(loadings)), end='\r')
        Wj = loadings[j]
        with Pool(njobs) as p:
            Hj = p.starmap(nnls, [(Wj, X[:, i].toarray().flatten()) for i in
                                  range(X.shape[1])])  # project each cell i of normalized data onto the current W
        Hj = np.vstack(Hj)  # Hj is projected data onto Wj
        projs.append(Hj)  # store projection
        Dj = Wj.dot(Hj.T)  # compute reconstructed data: Dj = Wj.Hj
        errors.append(
            mean_squared_error(D, Dj))  # compute mean squared error between original data D and reconstructed data Dj
    return errors, projs


def scale_W(W):
    """
    Scale the loadings. Divide each loading vector by its L2-norm.
    Parameters
    ----------
    W: array
    Loading matrix
    Returns
    -------
    Scaled loading matrix
    """
    norms = [np.linalg.norm(np.array(W[:, i]).flatten()) for i in
             range(W.shape[1])]  # compute the L2-norm of each feature
    return np.divide(W, norms)  # divide each factor by its L2-norm


def find_best_k(oNMFres, dataset_path, alpha=3, multiplier=3):
    """
    Find the optimal k by minimizing f(k;alpha,j)
    Parameters
    ----------
    oNMFres: dict
    The dict contains loading matrix, projections and errors of each factor number k
    dataset_path: path
    The path where the results will be stored
    alpha: float
    Power of polynomial
    multiplier: float
    Multiplies constant C in f(k)
    """
    print('Finding optimal factor number k...')
    errors = oNMFres['onmf']['errors']
    nfactors = oNMFres['onmf']['nfactors']

    # Normalize errors across all k
    errors_norm = np.divide(errors, np.max(errors))

    # Set all powers and scaling parameters to try
    alphas = [0.7, 0.9, 1, 2, 3, 4, 5]
    if alpha not in alphas:
        alphas.append(alpha)
        alphas = np.sort(alphas).tolist()

    basescales = []
    r_univ = range(1, 30)
    for i in range(len(alphas)):
        currscale = np.round(np.max(np.power(np.array(r_univ), alphas[i])) * 10)
        basescales.append(currscale)

    # Assemble f(k) values by sweeping over j and alpha
    fk_df = pd.DataFrame()
    minvals = []
    jrange = list(range(1, 8))
    if multiplier not in jrange:
        jrange.append(multiplier)
        jrange = np.sort(jrange).tolist()

    for i in range(len(alphas)):
        curralpha = alphas[i]
        currbasescale = basescales[i]
        currminvals = []
        for j in jrange:
            currscale = j * 1 / currbasescale  # increase j linearly
            curr_values = currscale * (np.power(np.array(nfactors), curralpha)) + np.array(errors_norm)
            currmin = nfactors[np.argwhere(curr_values == np.min(curr_values))[0][0]]

            curr_df = {'vals': np.append(curr_values, np.min(curr_values)),
                       'k': np.append(nfactors, currmin)}
            curr_df = pd.DataFrame(curr_df)
            curr_df['scale'] = currscale
            curr_df['a'] = curralpha
            curr_df['j'] = j
            curr_df['col'] = np.append(0 * curr_values, 1)
            fk_df = fk_df.append(curr_df, ignore_index=True)
            currminvals.append(currmin)
        minvals.append(currminvals)

    # Find best k given supplied alpha and multiplier:
    irow = alphas.index(alpha)
    icol = jrange.index(multiplier)
    idx_flat = irow * len(jrange) + icol
    bestk = minvals[irow][icol]

    # Save the factor loading matrix under best k
    idx = oNMFres['onmf']['nfactors'].index(bestk)
    np.save(os.path.join(dataset_path, 'loading.npy'), oNMFres['onmf']['loadings'][idx].T)
    # np.save(os.path.join(dataset_path, 'factor.npy'), oNMFres['onmf']['projs'][idx])

    # Plots
    # Plot1: Normalized error curve
    plt.scatter(nfactors, errors_norm, marker=".", s=100)
    plt.plot(nfactors, errors_norm, marker=".")
    plt.ylabel('Errors(normalized)')
    plt.savefig(os.path.join(dataset_path, 'oNMF_errors.png'), dpi=1000, bbox_inches="tight")
    plt.close()

    # Plot2: f(k; alpha, j) curves
    with sns.plotting_context('notebook', font_scale=1.7):
        g = sns.FacetGrid(fk_df, col="j", row="a", hue="col")
        g = (g.map(plt.scatter, "k", "vals", marker=".", s=400))  # .set(ylim=(0.03,0.055))
        g = (g.map(plt.plot, "k", "vals", marker="."))  # .set(ylim=(0.03,0.055))

        # Highlight the curve that uses the chosen parameters
        axes = g.axes.flatten()
        ax = axes[idx_flat]
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('magenta')
            spine.set_linewidth(2)

        plt.savefig(os.path.join(dataset_path, 'oNMF_fk_curves.png'), dpi=1000, bbox_inches="tight")
        plt.close()

    # Plot3: the optimal k under different alpha and j
    plt.figure(figsize=(5, 7))
    heat_map = sns.heatmap(minvals, annot=True, cmap='viridis')
    plt.yticks(plt.yticks()[0], alphas)
    plt.ylabel('alpha (a)')
    plt.xlabel('multiplier (j)')
    plt.yticks(rotation=0)
    plt.xticks(np.arange(len(jrange)) + 0.5, jrange)
    # highlight the k value selected in the data
    heat_map.add_patch(plt.Rectangle((icol, irow), 1, 1, fill=False, edgecolor='magenta', lw=2))
    plt.savefig(os.path.join(dataset_path, 'oNMF_optimal_k.png'), dpi=1000, bbox_inches="tight")
    plt.close()
    return bestk


def cal_factor(data, selected_loading, njobs=2):
    """
    Calculate factor-level expression for all cells given the selected loading.
    Parameters
    ----------
    data: AnnData
    Single-cell expression object
    selected_loading: array
    njobs: int
    """
    # recompute new projection based on chosen loading
    print('Projecting into selected loading...')
    with Pool(njobs) as p:
        H = p.starmap(nnls, [(selected_loading, data[:, i].toarray().flatten()) for i in
                             range(data.shape[1])])  # project each cell i of normalized data onto the current W
        proj = np.vstack(H)
    return proj


def onmf(data, dataset_name, ncells=2000, nfactors=list(range(5, 16)), nreps=2, niters=500, njobs=2, alpha=3,
         multiplier=3, random_state=53):
    """
    Perform oNMF. Compute factor loadings, factor-level expression (projections) and errors.
    And then select the optimal factor number k.
    Parameters
    ----------
    data: AnnData
    Single-cell expression object
    dataset_name: string
    The name of dataset to makedir (storing results)
    ncells: int
    Number of cells to use
    nfactors: int or list of ints
    Number(s) of factors to use
    nreps: int
    Number of repetitions to perform for each k in nfactors
    niters: int
    Maximum number of iterations to perform
    njobs: int
    Number of parallel jobs
    alpha: float
    The power of polynomial in selecting the optimal k
    multiplier: float
    The multiplies constant C in f(k) in selecting the optimal k
    """

    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)  # makedir
    dataset_path = os.path.join(dataset_name, 'oNMF')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)  # makedir to store oNMF results

    if type(nfactors) is int:
        nfactors = [nfactors]

    maxncells = data.shape[1]  # get total number of cells
    if 2 * ncells > maxncells:  # if ncells is larger than twice the number of cells
        ncells = int(np.floor(maxncells / 2))  # adjust number down

    # randomly select ncells and divide into training and cross-validation dataset
    np.random.seed(random_state)
    idx = np.random.choice(data.shape[1], 2 * ncells, replace=False)
    idx1 = idx[0:ncells]
    idx2 = idx[ncells:]

    print('Computing W matrices...')
    with Pool(njobs) as p:
        res = p.starmap(oNMF, [(data[:, idx1], x, niters) for x in
                               np.repeat(nfactors, nreps)])  # run ONMF in parallel for each possible k nreps times

    loadings = [scale_W(res[i][0]) for i in range(len(res))]  # scale the different feature spaces
    print('Computing reconstruction errors...')
    errors, projs = reconstruction_errors(data[:, idx2],
                                          loadings, njobs)  # compute the reconstruction errors

    # choose the best out of all replicates
    mlist = np.repeat(nfactors, nreps)

    bestofreps = []
    for i in range(len(nfactors)):
        curridx = np.where(mlist == nfactors[i])[0]
        currerrors = [errors[i] for i in curridx]
        bestidx = np.argmin(currerrors)
        bestofreps.append(curridx[bestidx])

    # select only the best of the replicate feature sets
    loadings = [loadings[i] for i in bestofreps]
    errors = [errors[i] for i in bestofreps]
    projs = [projs[i] for i in bestofreps]

    oNMFres = {}
    oNMFres['onmf'] = {}
    oNMFres['onmf']['loadings'] = loadings
    oNMFres['onmf']['projs'] = projs
    oNMFres['onmf']['errors'] = errors
    oNMFres['onmf']['nfactors'] = nfactors

    if len(nfactors) > 1:  # if more than one factor numbers in nfactors, find the optimal k
        find_best_k(oNMFres, dataset_path, alpha=alpha, multiplier=multiplier)
    else:
        np.save(os.path.join(dataset_path, 'loading.npy'), oNMFres['onmf']['loadings'][0].T)
        # np.save(os.path.join(dataset_path, 'factor.npy'), oNMFres['onmf']['projs'][0])

    selected_loading = (np.load(os.path.join(dataset_path, 'loading.npy'))).T
    selected_proj = cal_factor(data, selected_loading, njobs=2)
    np.save(os.path.join(dataset_path, 'factor.npy'), selected_proj)
    print('Finish oNMF. See the results in folder:{}'.format(dataset_path))


def plot_top_genes_loadings(gene_names, W, figsize=(8, 6), save_path=None, save=True):
    """
    Plot the top genes loadings for each factor and select the genes to do GSEA.
    Parameters
    ----------
    gene_names: list
    Gene names
    W: array
    Non-negative orthogonal loading matrix
    Returns
    -------
    The indices of top genes of each factor, which will be used to do GSEA
    """
    if save:
        if save_path is None:
            raise Exception("Please provide save_path to save the plot.")
    k = W.shape[1]
    stds = np.array(W.std(axis=0)).flatten()  # get the factor standard deviations
    stdfactor = 2  # factor to select genes
    genes_idx = np.array([])
    genes_loadings = np.array([])
    genes_do_go = {}
    for i, s in enumerate(stds):  # for each factor i and its respective standard deviations
        a = np.where(np.array(W[:, i]).flatten() > stdfactor * s)[0]  # get indices of top genes
        sub = np.array(W[a, i]).flatten()  # grab loadings of top genes for factor i
        a = a[np.argsort(sub)[::-1]]  # sort subset by loadings
        genes_do_go[str(i)] = a
        genes_idx = np.concatenate([genes_idx, a])  # concatenate gene indices
        genes_loadings = np.concatenate([genes_loadings, np.sort(sub)[::-1]])
    genes_idx = genes_idx.astype(int)

    # make list unique
    s = set()
    l = []
    vals = []
    for i in range(len(genes_idx)):
        x = genes_idx[i]
        currval = genes_loadings[i]
        if x in s:
            # find the existing loading value and compare it to the new loading value
            origidx = np.argwhere(l == x)[0][0]
            origval = vals[origidx]
            if origval < currval:
                # remove the origval and gene name in the list and append the new val and gene name
                l.remove(x)
                vals.remove(origval)
                l.append(x)
                vals.append(currval)

        if x not in s:
            s.add(x)
            l.append(x)
            vals.append(currval)

    genes_idx = np.array(l)[::-1]
    keptgenes = [gene_names[i] for i in genes_idx]

    # define gene font sizes
    geneFS = 3  # (default)
    if len(genes_idx) > 200:
        geneFS = 2
    if len(genes_idx) > 400:
        geneFS = 1
    if len(genes_idx) > 600:
        geneFS = 0.5

    mtx = W[genes_idx, :]  # subset feature space with the top genes for the features

    with plt.rc_context({'figure.figsize': figsize, 'font.sans-serif': ['Arial']}):
        ar = 2.5 * k / len(genes_idx)  # define aspect ratio
        plt.imshow(mtx, cmap='magma', aspect=ar, interpolation='none')  # create heatmap
        plt.ylabel('Genes')
        plt.ylim(-0.5, len(genes_idx) - 0.5)
        plt.yticks(np.arange(len(genes_idx)), keptgenes, fontsize=geneFS)
        # plt.yticks(np.arange(len(genes_idx)), [''] * len(genes_idx))
        plt.xticks(np.arange(k), np.arange(k) + 1)
        plt.xlabel('Factor')
        plt.colorbar(pad=0.02)
        if save:
            plt.savefig(os.path.join(save_path, 'Gene_loadings.svg'), bbox_inches='tight', dpi=1000)
        plt.show()
    return genes_do_go
