from email.headerregistry import Group
import time
import math
import numpy as np
from collections import Counter
import utils 
import os
import pickle
import random
from multiprocessing import Pool
from functools import partial
from math import gcd as bltin_gcd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

d_id = '1'

if not os.path.exists(utils.RESULT_PATH):
    os.makedirs(utils.RESULT_PATH)
if not os.path.exists(utils.PLOTS_PATH):
    os.makedirs(utils.PLOTS_PATH)
""" choose dataset """
#d_id = input("input evaluation dataset: 1. Enron 2. Lucene 3.WikiPedia ")
dataset_name = ''
number_queries_per_period = 1000
observed_period = 8
target_period = 10
adv_observed_offset = 10
if d_id=='1':
    dataset_name = 'Enron'
elif d_id=='2':
    dataset_name = 'Lucene'  
    observed_period = 16
elif d_id=='3':
    dataset_name = 'Wiki'
    number_queries_per_period = 5000
    observed_period = 32
else:
    raise ValueError('No Selected Dataset!!!')

#plot_figure(dataset_name)
""" read data """
with open(os.path.join(utils.DATASET_PATH,"{}_doc.pkl".format(dataset_name.lower())), "rb") as f:
    docu = pickle.load(f)
    f.close()
with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
    kw_dict = pickle.load(f)
    f.close()
chosen_kws = list(kw_dict.keys())
with open(os.path.join(utils.DATASET_PATH, "{}_wl_v_off.pkl".format(dataset_name.lower())), "rb") as f:
    real_size, real_length, offset_of_Decoding = pickle.load(f)#_after_padding_2
    f.close()
exp_times = 10
offset_of_Decoding_list = [offset_of_Decoding]*exp_times


class Attack: 
    def __init__(self, doc, min_file_size, max_file_size, ShieldAlpha, VIAt, chosen_kws, observed_queries, target_queries, kws_leak_percent, trend_matrix_norm, real_size, real_length, offset_of_Decoding):
        self.real_tag = {}
        self.recover_tag = {}

        self.real_group = {}
        self.recover_group = {}
        self.cluster_acc = 0
        self.gamma = (int) (len(chosen_kws)/2)

        self.setup_bandwidth = 0
        self.update_bandwidth = 0
        self.query_bandwidth = 0

        self.doc = doc

        self.time = 0
        self.inject_time = 0        
        self.recover_queries_num = 0
        self.total_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0

        self.bandwidth_setup = 0 #setup database
        self.bandwidth_update = 0 #a file contains multiple keywords, BVA, BVMA ;no padding of Decoding, impractical of SR.
        self.bandwidth_query = 0 #return No. of files
        #self.bandwidth_query_after_inj = 0 #return No. of files

        self.kws_leak_percent = kws_leak_percent
        self.chosen_kws = chosen_kws
        self.target_queries = target_queries
        self.observed_queries = observed_queries
        self.trend_matrix_norm = trend_matrix_norm
        
        self.ShieldAlpha = ShieldAlpha
        self.VIAt = VIAt

        self.size_without_padding, self.length_without_padding = real_size, real_length 
        self.size_after_setup_padding, self.length_after_setup_padding = {}, {}
        self.size_after_injection_padding, self.length_after_injection_padding = {}, {}
        self.min_file_size, self.max_file_size = min_file_size, max_file_size 
        self.injection_length_without_padding = {}
        """
        baseline phase
        """
        #self.observed_size, self.max_observed_size, self.observed_length = self.get_baseline_observed_size_and_length(real_size, real_length)
        """
        get offset of Decoding
        """
        self.offset = offset_of_Decoding
        self.Group = self.Group_cluster()
        self.Col = self.get_kws_column()
        self.injection_parameter = self.get_injection_parameter()

    #def is_coprime(a, b):
    #    return bltin_gcd(a, b) == 1

    def get_injection_parameter(self):
        b = math.ceil(len(chosen_kws)/(2*self.ShieldAlpha))
        inj_P = []
        inj_P.append(b)
        b += 1
        while(len(inj_P)<self.VIAt):
            flag = False
            while(not flag):
                lt = 0
                while(lt<len(inj_P)):
                    #print(inj_P[lt])
                    #print(b)
                    if bltin_gcd(inj_P[lt], b)!=1:
                        break
                    lt += 1
                if lt == len(inj_P):
                    flag = True
                    inj_P.append(b) 
                b += 1     
        return inj_P

    def get_size_and_length_after_injection_padding(self, injection_length, injection_size):
        """
        dict: {[keyword, inject length]} // {[keyword, inject size]}
        """
        self.size_after_injection_padding = {}
        self.length_after_injection_padding = {}
        for Gp in self.Group:
            max_length_of_each_cluster = 0
            for k in Gp:
                if injection_length[k]>max_length_of_each_cluster:
                    max_length_of_each_cluster=injection_length[k]
            for k in Gp:
                self.size_after_injection_padding[k] = self.size_after_setup_padding[k]
                self.length_after_injection_padding[k] = self.length_after_setup_padding[k]

                self.size_after_injection_padding[k] += injection_size[k]
                if max_length_of_each_cluster - injection_length[k]>20:
                    self.size_after_injection_padding[k] += (max_length_of_each_cluster - injection_length[k])*random.randint(self.min_file_size, self.max_file_size)
                else:
                    for _ in range(max_length_of_each_cluster - injection_length[k]):
                        self.size_after_injection_padding[k] += random.randint(self.min_file_size, self.max_file_size)
                
                self.length_after_injection_padding[k] += max_length_of_each_cluster


    def get_size_and_length_after_setup_padding(self):
        """
        Padding of setup phase
        """
        self.size_after_setup_padding = {}
        self.length_after_setup_padding = {}
        for Gp in self.Group:
            max_length_of_each_cluster = 0
            for k in Gp:
                if self.length_without_padding[k]>max_length_of_each_cluster:
                    max_length_of_each_cluster=self.length_without_padding[k]
            for k in Gp:
                self.size_after_setup_padding[k] = self.size_without_padding[k]
                for _ in range(max_length_of_each_cluster - self.length_without_padding[k]):
                    self.size_after_setup_padding[k] += random.randint(self.min_file_size, self.max_file_size)
                self.length_after_setup_padding[k] = max_length_of_each_cluster

    def get_kws_column(self):
        Col = []
        for j in range(len(self.Group[0])): #self.Group[0]
            CW = []
            for i in range(len(self.Group)):
                if j<len(self.Group[i]):
                    CW.append(self.Group[i][j])
            Col.append(CW)
        #print(Col)
        return Col

    def Group_cluster(self):
        """
        Caching cluster
        """
        with open(os.path.join(utils.DATASET_PATH, "{}_wl_v_off_SampleDataOfShieldDB.pkl".format(dataset_name.lower())), "rb") as f:
            new_keyword_identifier = pickle.load(f)#_after_padding_2
            f.close()
        #print(len(new_keyword_identifier))
        Group = []
        subgroup = []
        count = 0
        for i in range(len(new_keyword_identifier)):
            if count==self.ShieldAlpha:
                Group.append(subgroup)
                count = 0
                subgroup = []
            subgroup.append(new_keyword_identifier[i])
            count += 1
        if len(subgroup)!=0:
            Group.append(subgroup)
        return Group


    def Loc_query_to_group(self, query):
        recover_g= -1
        real_g = -2
        CA = []
        for i in range(len(self.Group)):
            if query in self.Group[i]:
                real_g = i
            if len(self.Group[i])!=0 and self.length_after_injection_padding[self.Group[i][0]] == self.length_after_injection_padding[query]:
                CA.append(i)
                #recover_g = i
        recover_g = CA
        return real_g, recover_g
        
    def Location_query_to_group(self):
        self.get_size_and_length_after_setup_padding()
        #observed_size_in_setup, max_observed_size_in_setup, observed_length_in_setup = self.get_baseline_observed_size_and_length()
        """
        injection: get self.length_after_injection_padding, self.length_after_injection_padding
        """
        self.OptimizedVIA_inject()
        self.recover_queries_num = 0
        self.total_queries_num = 0
        #for i in range(40):
        #    print(self.length_after_setup_padding[self.Group[i][0]])
        #_, _, observed_length = self.get_baseline_observed_size_and_length()
        self.recover_group = {}
        self.real_group = {}
        for i_week in self.target_queries:
            for query in i_week:
                self.real_group[query], self.recover_group[query] = self.Loc_query_to_group(query)
        #for k in self.recover_group.keys():
                #print(query)
                #print(self.Group[self.real_group[query]])
                if random.choice(self.recover_group[query]) == self.real_group[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
        self.cluster_acc = self.recover_queries_num/self.total_queries_num
   

    def get_baseline_observed_size_and_length(self):
        """
        observe size and length in baseline phase
        """
        observed_size = {}
        observed_length = {}
        max_observed_size = 0
        for i_week in self.target_queries:
            for query in i_week:
                observed_size[query] = self.size_after_setup_padding[query]
                observed_length[query] = self.length_after_setup_padding[query]
                if max_observed_size < observed_size[query]:
                    max_observed_size = observed_size[query]
        return observed_size, max_observed_size, observed_length

    def OptimizedVIA_main(self):
        self.total_queries_num = 0
        self.recover_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0
        """
        Setup: get self.length_after_setup_padding, self.length_after_setup_padding
        """
        self.get_size_and_length_after_setup_padding()
        observed_size_in_setup, max_observed_size_in_setup, observed_length_in_setup = self.get_baseline_observed_size_and_length()
        """
        injection: get self.length_after_injection_padding, self.length_after_injection_padding
        """
        self.OptimizedVIA_inject()
        """
        recovery
        """
        s = time.time()
        self.OptimizedVIA_recover(observed_size_in_setup)
        e = time.time()
        self.time = e-s
        # print("BVARecoeryTime:{}".format(self.time))
        self.accuracy = self.recover_queries_num/self.total_queries_num
        # print("BVARecoeryRer:{}".format(self.accuracy))
    def OptimizedVIA_recover(self, observed_size_in_setup):
        self.real_tag = {}
        self.recover_tag = {}

        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                CW = []
                for group in self.recover_group[query]:
                    for kw_id in self.Group[group]: 
                        CW.append(kw_id)
                #print(query)
                #print(CW)
                #print(self.recover_group[query])
                for group in self.recover_group[query]:
                    flag_found = False
                    #if query not in self.Group[group]:
                    #    print(self.Group[group])
                    for query_in_setup in observed_size_in_setup.keys(): ###############self.Group[group]
                        if query in self.size_after_injection_padding.keys(): #and kw_id in observed_size_in_setup.keys(): 
                            for ht in range(len(self.injection_parameter)):
                                if (self.size_after_injection_padding[query] - observed_size_in_setup[query_in_setup]) % self.injection_parameter[ht] == 0:
                                    v = (int) ((self.size_after_injection_padding[query] - observed_size_in_setup[query_in_setup]) / self.injection_parameter[ht])
                                    
                                    if v>=0 and v<len(self.Group) and ht<len(self.Group[v]) and self.Group[v][ht] in CW:
                                        #print("query:{}".format(query))
                                        #print("V:{}".format(v))
                                        #print("self.Group[v]:{}".format(self.Group[v]))
                                        #print("ht:{}".format(ht))
                                        #print("self.Group[v][ht]:{}".format(self.Group[v][ht]))
                                        self.recover_tag[query] = (self.Group[v][ht])
                                    #if self.recover_tag[query] in self.Col[ht]: self.recover_tag[query]
                                        flag_found = True
                                        break
                            if flag_found:
                                break
                    if flag_found:
                        break
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def OptimizedVIA_inject(self):
        """
        generate injected doc
        """
        """
        size after injection
        """
        self.total_inject_length = 0
        injection_length = {}
        injection_size = {}
        for kw in range(len(chosen_kws)):
            injection_length[kw] = 0
            injection_size[kw] = 0

        for j in range(len(self.injection_parameter)):
            CW = self.Col[j]
            #print(CW)
            num_injection_doc = math.ceil(np.log2(len(CW)))
            self.total_inject_length += num_injection_doc
            size_each_doc = []
            if num_injection_doc >= 1:
                size_each_doc.append(self.injection_parameter[j])
                #self.total_inject_size += size_each_doc[0]
            if num_injection_doc >= 2:
                for i in range(1, num_injection_doc):
                    size_each_doc.append(size_each_doc[i-1] + size_each_doc[i-1])
                    #self.total_inject_size += size_each_doc[i]

            for k in range(len(CW)):
                #injection_length[CW[k]] = 0
                #injection_size[CW[k]] = 0
                for num_ind in range(num_injection_doc):
                    if ((k >> num_ind) & 1) == 1 :
                        injection_length[CW[k]] += 1
                        injection_size[CW[k]] += size_each_doc[num_ind]
                        #real_size_after_injection[kws_ind] += size_each_doc[num_ind]       
        #print(injection_length.keys())
        self.injection_length_without_padding = injection_length
        self.get_size_and_length_after_injection_padding(injection_length, injection_size)



def multiprocess_worker(kw_dict, chosen_kws, docu, aa, tt, adv_observed_offset, observed_period, target_period, number_queries_per_period, real_size, real_length, offset_of_Decoding):

    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), 260, adv_observed_offset)
    begin_time = random.randint(0, len(trend_matrix[0])-observed_period-target_period-1)
    observed_queries = utils.generate_queries(trend_matrix[:, begin_time:begin_time+observed_period], 'real-world', number_queries_per_period)
    target_queries = utils.generate_queries(trend_matrix[:, begin_time+observed_period:begin_time+observed_period+target_period], 'real-world', number_queries_per_period)

    min_file_size, max_file_size, _ = utils.get_file_size(docu)
    sdb = Attack(docu, min_file_size, max_file_size, aa, tt, chosen_kws, observed_queries, target_queries, 1, trend_matrix, real_size, real_length, offset_of_Decoding)
    sdb.Location_query_to_group()

    sdb.OptimizedVIA_main()
    BVA_acc = sdb.accuracy
    # print(sdb.accuracy)
    return [BVA_acc, sdb.total_inject_length]

def plot_figure_T(ABVIA_acc, Injection_length):
    labels = [r'1', r'2', r'4', r'8', r'16', r'32', r'64', r'128']

    plt.clf()
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], 
    "font.size":25,
    "lines.markersize":17})

    _, ax = plt.subplots()

    l1, = ax.plot(labels, np.mean(ABVIA_acc, axis=1), label = 'Rer', color="lightgreen",markeredgecolor='green',marker="o") # '--g'
    ax.set_xlabel("t")
    ax.set_ylabel("Recovery rate")
    ax.set_ylim(0.0, 1)
    #ax.legend()
    ax2 = ax.twinx()
    l2, = ax2.plot(labels, np.mean(Injection_length, axis=1), label = 'ILen', color="lightblue",markeredgecolor='blue',marker="X") # '--g'
    ax2.set_ylabel("Injection length")
    #ax2.legend()
    ax2.set_ylim(0.0, 1000)

    plt.legend(handles=[l1, l2], labels=['Rer', 'ILen'], loc=0)
    plt.grid()
    plt.savefig(utils.PLOTS_PATH + '/' + 'ABVIAShieldDBEnronT.pdf', bbox_inches = 'tight', dpi = 600)
    plt.show() 

def plot_figure_Alpha(ABVIA_acc, Injection_length):
    labels = ['2', '8', '32', '128']

    plt.clf()
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    #"font.family": "serif",
    #"font.serif": ["Times"], #注意这里是Times，不是Times New Roman
    "font.size":26,
    "lines.markersize":17})

    fig, ax = plt.subplots()

    l1, = ax.plot(labels, np.mean(ABVIA_acc, axis=1), label = 'Rer', color="lightgreen",markeredgecolor='green',marker="o") # '--g'
    ax.set_xlabel(r'$alpha$')
    ax.set_ylabel("Recovery rate")
    ax.set_ylim(0.65, 1)
    ##ax.legend(loc=1)
    ax2 = ax.twinx()
    l2, = ax2.plot(labels, np.mean(Injection_length, axis=1), label = 'ILen', color="lightblue",markeredgecolor='blue',marker="X") # '--g'
    ax2.set_ylabel("Injection length")
    ax2.set_ylim(0.0, 700)

    #ax2.legend(loc=2)
    plt.legend(handles=[l1, l2], labels=['Rer', 'ILen'], loc=0)

    plt.grid()
    plt.savefig(utils.PLOTS_PATH + '/' + 'ABVIAShieldDBEnronAlpha.pdf', bbox_inches = 'tight', dpi = 600)
    plt.show() 


if __name__=='__main__': 
   

    BVA_acc_different_t = []
    Injection_length_different_t = []
 
    ShieldDBalpha = 128 
    ParameterT = [1, 2, 4, 8, 16, 32, 64, 128]
    pbar = tqdm(total=len(ParameterT))
    loop = 0

    for aa in ParameterT:
        BVA_tmp_acc = []
        Injection_tmp_length = []

        partial_function = partial(multiprocess_worker, kw_dict, chosen_kws, docu, ShieldDBalpha, aa, adv_observed_offset, observed_period, target_period, number_queries_per_period, real_size, real_length)
        with Pool(processes=exp_times) as pool:
            for result in pool.map(partial_function, offset_of_Decoding_list):
                BVA_tmp_acc.append(result[0])
                Injection_tmp_length.append(result[1])

        BVA_acc_different_t.append(BVA_tmp_acc)
        Injection_length_different_t.append(Injection_tmp_length)
        pbar.update(math.ceil((loop+1)/len(ParameterT)))
        loop += 1
    pbar.close()
    with open(os.path.join(utils.RESULT_PATH, 'ShieldDBABVIADifferentT.pkl'), 'wb') as f:
        pickle.dump((BVA_acc_different_t, Injection_length_different_t), f)
        f.close()
    print(BVA_acc_different_t)
    print(Injection_length_different_t)


    BVA_acc_different_alpha = []
    Injection_length_different_alpha = []
    ShieldDBAlpha = [2, 8, 32, 128]
    pbar = tqdm(total=len(ShieldDBAlpha))
    loop = 0
    for aa in ShieldDBAlpha:
        BVA_tmp_acc = []
        Injection_tmp_length = []

        partial_function = partial(multiprocess_worker, kw_dict, chosen_kws, docu, aa, aa, adv_observed_offset, observed_period, target_period, number_queries_per_period, real_size, real_length)
        with Pool(processes=exp_times) as pool:
            for result in pool.map(partial_function, offset_of_Decoding_list):
                BVA_tmp_acc.append(result[0])
                Injection_tmp_length.append(result[1])

        BVA_acc_different_alpha.append(BVA_tmp_acc)
        Injection_length_different_alpha.append(Injection_tmp_length)
        pbar.update(math.ceil((loop+1)/len(ShieldDBAlpha)))
        loop += 1
    pbar.close()
    with open(os.path.join(utils.RESULT_PATH, 'ShieldDBABVIADifferentAlpha.pkl'), 'wb') as f:
        pickle.dump((BVA_acc_different_alpha, Injection_length_different_alpha), f)
        f.close()
    print(BVA_acc_different_alpha)
    print(Injection_length_different_alpha)

    plot_figure_Alpha(BVA_acc_different_alpha, Injection_length_different_alpha)
    plot_figure_T(BVA_acc_different_t, Injection_length_different_t)