#! /usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn
import sys
import os.path
import warnings
import pymc3 as pm
from scipy.stats import norm, uniform, gamma, poisson, beta, bernoulli
from numpy.random import multinomial,dirichlet

warnings.simplefilter('ignore')

from scipy import stats
# from matplotlib import dates,ticker

# from pandas.io.json import json_normalize
# from bson.objectid import ObjectId

# from pymongo import MongoClient
# db_client = MongoClient()


def read_data(filepath):
    df = pd.read_csv(filepath,sep=' ')
    df["content"] = df["content"].apply(lambda x: [ int(xx) for xx in x.split(',')])
    return df,len(df["user"].value_counts()),len(df["video"].value_counts())

def run_model(df,M,N):
    print "Start to parse N = %d videos from M = %d users"%(N,M)
    N_user_type = 2
    N_senti_type = 3
    N_iter = 100
    burnin = 50

    prior_alpha_0 = np.array([2.0]*N_user_type)
    prior_alpha_1 = np.array([2.0]*N_senti_type)
    # prior_alpha_0[0] = 50

    # lambdaR = np.array([[0]*M]*N)
    # AlphaS = np.array([[0]*M]*N)
    # PossionR = np.array([[0]*M]*N)
    # MultiS = np.array([[0]*M]*N)

    v_beta = np.array([0.0]*N)
    truth = np.array([0]*N)
    diri = np.array([[0.0]*N_user_type]*M)
    user_type = np.array([0]*M)


    lambdaR = np.array([[0.0]*2]*N_user_type)
    AlphaS = np.array([[[0.0]*N_senti_type]*2]*N_user_type)
    PossionR = np.array([[0.0]*2]*N_user_type)
    MultiS = np.array([[0.0]*2]*N_user_type)



    a = 1.0; b = 1.0;
    # INITIATE
    for video_id in range(N):
        v_beta[video_id] = beta.rvs(1,1) # process prior from users
        truth[video_id] = bernoulli.rvs(v_beta[video_id])

    for user_id in range(M):
        diri[user_id] = dirichlet(prior_alpha_0)
        user_type[user_id] = np.where(multinomial(1,diri[user_id],size=1)==1)[1][0]

    for user_type_id in range(N_user_type):
        for video_type_id in range(2):
            lambdaR[user_type_id][video_type_id] = gamma.rvs(a,scale=1./b)
            # PossionR[user_type_id][video_type_id] = poisson.rvs("possionR_%d_%d"%(user_type_id,video_type_id),mu=lambdaR[user_type_id][video_type_id],observed=R_obs)
            AlphaS[user_type_id][video_type_id] = dirichlet(prior_alpha_1)
            # MultiS[user_type_id][video_type_id] = pm.Multinomial("multiS_%d_%d"%(user_type_id,video_type_id),n=1,p=AlphaS[user_type_id][video_type_id],shape=(N_senti_type,),observed=S_obs)
    # lambdaR[0][1] = gamma.rvs(50,scale=1./10)
    # lambdaR[0][0] = gamma.rvs(1,scale=1./10)

    re_user_type = np.array([[1.0]*N_user_type]*M)
    re_truth = np.array([[1.0]*2]*N)
    full_truth_table = np.array([[[0.0,0.0]]*N]*M)
    # Start Sample
    tmp_user_type = np.array([[0.0]*N_user_type]*M)
    tmp_truth = np.array([[0.0]*2]*N)
    for n in range(N_iter):

        # R_obs = ([[[]]*2]*N_user_type)
        R_obs = [[ [],[] ] for _ in range(N_user_type)]
        S_obs = np.array([[[0.0]*N_senti_type]*2]*N_user_type)
        for index,row in df.iterrows():
            for value in row["content"]:
                if row["type"] == 'r':
                        R_obs[user_type[row["user"]]][truth[row["video"]]].append(value)
                elif row["type"] == 'e':
                        S_obs[user_type[row["user"]]][truth[row["video"]]][value]+=1

        # print "R_obs\n",R_obs
        # print "S_obs\n",S_obs

        for user_type_id in range(N_user_type):
            for video_type_id in range(2):
                lambdaR[user_type_id][video_type_id] = gamma.rvs(a+sum(R_obs[user_type_id][video_type_id]),scale=1.0/(b+len(R_obs[user_type_id][video_type_id])))
                AlphaS[user_type_id][video_type_id] = dirichlet(prior_alpha_1 + S_obs[user_type[row["user"]]][truth[row["video"]]])

        # print "lambdaR\n",lambdaR
        # print "AlphaS\n",AlphaS
        # lambdaR[0][1] = gamma.rvs(50+sum(R_obs[0][1]),scale=1./(10 + len(R_obs[0][1])))
        # lambdaR[0][0] = gamma.rvs(1+sum(R_obs[0][0]),scale=1./(10 + len(R_obs[0][0])))

        tmp_user_type = np.array([[0.0]*N_user_type]*M)
        tmp_user_type_self = np.array([0.0]*M)
        for user_id in range(M):
            for video_id in range(N):
                tmp_max = 0.0
                tmp = np.array([0.0]*N_user_type)
                new_user_type = user_type[user_id]
                for user_type_id in range(N_user_type):
                    try:
                        t_r_obs = df[(df['user']==user_id)&(df['video']==video_id)&(df['type']=='r')].ix[:,'content'].values[0]
                        t_s_obs = df[(df['user']==user_id)&(df['video']==video_id)&(df['type']=='e')].ix[:,'content'].values[0]
                    except:
                        pass


                    tmp[user_type_id] = (sum(t_r_obs))*np.log(lambdaR[user_type_id][truth[video_id]])\
                        + (- sum(t_r_obs))*np.log(lambdaR[user_type[user_id]][truth[video_id]])\
                        - (len(t_r_obs))*lambdaR[user_type_id][truth[video_id]]\
                        - (- len(t_r_obs))*lambdaR[user_type[user_id]][truth[video_id]]

                    for s_ in t_s_obs:
                        tmp[user_type_id] += np.log(AlphaS[user_type_id][truth[video_id]][s_])\
                            - np.log(AlphaS[user_type[user_id]][truth[video_id]][s_])

                    tmp_user_type[user_id][user_type_id] += tmp[user_type_id]

                    if user_type_id == user_type[user_id]:
                        tmp_user_type_self[user_id] += (sum(t_r_obs))*np.log(lambdaR[user_type_id][truth[video_id]])\
                            - (len(t_r_obs))*lambdaR[user_type_id][truth[video_id]]
                        for s_ in t_s_obs:
                            tmp_user_type_self[user_id] += np.log(AlphaS[user_type_id][truth[video_id]][s_])


        for user_id in range(M):
            Prob_user_type = np.exp(tmp_user_type[user_id] + tmp_user_type_self[user_id])
            if max(tmp_user_type[user_id]) > 0:
                new_user_type = np.argmax(tmp_user_type[user_id])
                if tmp_user_type[user_id][new_user_type] > 0\
                        and Prob_user_type[new_user_type]/sum(Prob_user_type)\
                        > re_user_type[user_id][user_type[user_id]]/sum(re_user_type[user_id]):
                    user_type[user_id] = new_user_type

            if n > burnin:
                re_user_type[user_id][user_type[user_id]] += 1

        tmp_truth = np.array([[0.0]*2]*N)
        tmp_truth_self = np.array([0.0]*N)
        for video_id in range(N):
            for user_id in range(M):
                tmp_max = 0.0
                tmp = np.array([0.0]*2)
                new_video_type = truth[video_id]
                for video_type_id in range(2):
                    try:
                        t_r_obs = df[(df['user']==user_id)&(df['video']==video_id)&(df['type']=='r')].ix[:,'content'].values[0]
                        t_s_obs = df[(df['user']==user_id)&(df['video']==video_id)&(df['type']=='e')].ix[:,'content'].values[0]
                    except:
                        pass

                    tmp[video_type_id] = (sum(t_r_obs))*np.log(lambdaR[user_type[user_id]][video_type_id])\
                        + (- sum(t_r_obs))*np.log(lambdaR[user_type[user_id]][truth[video_id]])\
                        - (len(t_r_obs))*lambdaR[user_type[user_id]][video_type_id]\
                        - (- len(t_r_obs))*lambdaR[user_type[user_id]][truth[video_id]]

                    for s_ in t_s_obs:
                        tmp[video_type_id] += np.log(AlphaS[user_type[user_id]][video_type_id][s_])\
                            - np.log(AlphaS[user_type[user_id]][truth[video_id]][s_])

                    tmp_truth[video_id][video_type_id] += tmp[video_type_id]

                    if video_type_id == truth[video_id]:
                        tmp_truth_self[video_id] += (sum(t_r_obs))*np.log(lambdaR[user_type[user_id]][video_type_id])\
                            - (len(t_r_obs))*lambdaR[user_type[user_id]][video_type_id]
                        for s_ in t_s_obs:
                            tmp_truth_self[user_id] += np.log(AlphaS[user_type[user_id]][video_type_id][s_])


        for video_id in range(N):
            Prob_truth = np.exp(tmp_truth[video_id] + tmp_truth_self[video_id])
            if max(tmp_truth[video_id]) > 0:
                new_video_type = np.argmax(tmp_truth[video_id])
                if tmp_truth[video_id][new_video_type] > 0\
                        and Prob_truth[new_video_type]/sum(Prob_truth)\
                        > re_truth[video_id][truth[video_id]]/sum(re_truth[video_id]):
                    # print "change occurs***********",video_id,truth[video_id],new_video_type,\
                    #     Prob_truth[new_video_type]/sum(Prob_truth),re_truth[video_id][truth[video_id]]/sum(re_truth[video_id])
                    # print "Lambda",lambdaR,"AlphaS",AlphaS
                    truth[video_id] = new_video_type
                # else:
                #     print "change tried***********",video_id,truth[video_id],new_video_type,\
                #         Prob_truth[new_video_type]/sum(Prob_truth),re_truth[video_id][truth[video_id]]/sum(re_truth[video_id])


            if n > burnin:
                re_truth[video_id][truth[video_id]] += 1

        print "\nIteration #%d of %d"%(n,N_iter),truth,user_type

    print "#####################RESULT###################\n\nVideo Type"
    for i in range(N):
        print i,np.argmax(re_truth[i]),np.max(re_truth[i])/float(sum(re_truth[i]))

    print "\nUser Type"
    for i in range(M):
        print i,np.argmax(re_user_type[i]),np.max(re_user_type[i])/float(sum(re_user_type[i]))

    print "\nR mean:\n",lambdaR
    print "\nS mean:\n",AlphaS
    discern_truth = [0.0,0.0]
    for user_id in range(M):
        for video_id in range(N):
            discern_truth[np.argmax(full_truth_table[user_id][video_id])] += lambdaR[np.argmax(re_user_type[user_id])][np.argmax(full_truth_table[user_id][video_id])]
            discern_truth[1-np.argmax(full_truth_table[user_id][video_id])] += lambdaR[np.argmax(re_user_type[user_id])][1-np.argmax(full_truth_table[user_id][video_id])]

    print "\nVideo Label 0 denotes a",discern_truth[0] > discern_truth[1],"video"

    return truth








if __name__ == "__main__":
    np.random.seed(12423)
    data_df,N_user,N_video = read_data("./test.csv")
    run_model(data_df,N_user,N_video)


