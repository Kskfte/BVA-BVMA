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
    def __init__(self, doc, min_file_size, max_file_size, ShieldAlpha, chosen_kws, observed_queries, target_queries, kws_leak_percent, trend_matrix_norm, real_size, real_length, offset_of_Decoding):
        self.real_tag = {}
        self.recover_tag = {}

        self.real_group = {}
        self.recover_group = {}
        self.cluster_acc = 0
        self.gamma = (int) (len(chosen_kws)/2)

        self.bandwidth_setup = 0 #setup database
        self.size_setup = 0
        self.bandwidth_static_query = 0 #return No. of files for query before injection
        self.size_static_query = 0 # No. of returned files per query
        
        self.bandwidth_injection = 0 #a file contains multiple keywords, BVA, BVMA ;no padding of Decoding, impractical of SR.
        self.size_injection = 0
        self.bandwidth_dynamic_query = 0 #return No. of files for query after injection
        self.size_dynamic_query = 0 # No. of returned files per query

        self.doc = doc

        self.time = 0
        self.inject_time = 0        
        self.recover_queries_num = 0
        self.total_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0

        #self.bandwidth_setup = 0 # setup database
        #self.bandwidth_update = 0 # a file contains multiple keywords, BVA, BVMA ;no padding of Decoding, impractical of SR.
        #self.bandwidth_query = 0 # return No. of files

        self.kws_leak_percent = kws_leak_percent
        self.chosen_kws = chosen_kws
        self.target_queries = target_queries
        self.observed_queries = observed_queries
        self.trend_matrix_norm = trend_matrix_norm
        
        self.ShieldAlpha = ShieldAlpha
  
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
        #print(self.Group)
    
    def get_bandwidth(self):
        """
        Increased No. of file compared with no padding.
        """
        self.get_size_and_length_after_setup_padding()
        number_of_padding_setup = 0
        for k in self.length_after_setup_padding.keys():
            tmp = self.length_after_setup_padding[k]-self.length_without_padding[k]
            if tmp>number_of_padding_setup:
                number_of_padding_setup = tmp
        #print("keyword: {}".format(kkk))
        #print("length_without_padding: {}".format(self.length_without_padding[kkk]))
        #print("length_after_setup_padding: {}".format(self.length_after_setup_padding[kkk]))
        self.size_setup = number_of_padding_setup + len(self.doc)
        self.bandwidth_setup = number_of_padding_setup/len(self.doc)

        self.BVMA_SP_main('real-world')
        self.size_static_query = (int) (sum(self.length_after_setup_padding.values())/len(chosen_kws))
        self.bandwidth_static_query = sum(self.length_after_setup_padding.values()) / sum(self.length_without_padding.values()) - 1
        # print((int) (sum(self.length_after_setup_padding.values())/len(chosen_kws)))
        # print((int) (sum(self.length_without_padding.values())/len(chosen_kws)))

        self.size_dynamic_query = (int) (sum(self.length_after_injection_padding.values())/len(chosen_kws))
        self.bandwidth_dynamic_query = sum(self.length_after_injection_padding.values()) / (sum(self.length_without_padding.values()) + sum(self.injection_length_without_padding.values())) - 1
        
        number_of_padding_injection = 0
        for k in self.length_after_injection_padding.keys():
            tmp = self.length_after_injection_padding[k]-self.injection_length_without_padding[k]-self.length_after_setup_padding[k]
            if tmp>number_of_padding_injection:
                number_of_padding_injection = tmp
        self.bandwidth_injection = number_of_padding_injection/math.ceil(math.log2(len(chosen_kws)))
        self.size_injection = (number_of_padding_injection + math.ceil(math.log2(len(chosen_kws))))

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

    def Group_cluster(self):
        """
        Caching cluster
        """
        with open(os.path.join(utils.DATASET_PATH, "{}_wl_v_off_SampleDataOfShieldDB.pkl".format(dataset_name.lower())), "rb") as f:
            new_keyword_identifier = pickle.load(f)#_after_padding_2
            f.close()
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
        self.BVA_inject()
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
                if random.choice(self.recover_group[query]) == self.real_group[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
        self.cluster_acc = self.recover_queries_num/self.total_queries_num
        # print(self.cluster_acc)

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

    def BVA_main(self, gamma):
        self.total_queries_num = 0
        self.recover_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0
        self.gamma = gamma
        """
        Setup: get self.length_after_setup_padding, self.length_after_setup_padding
        """
        self.get_size_and_length_after_setup_padding()
        observed_size_in_setup, max_observed_size_in_setup, observed_length_in_setup = self.get_baseline_observed_size_and_length()
        """
        injection: get self.length_after_injection_padding, self.length_after_injection_padding
        """
        self.BVA_inject()
        """
        recovery
        """
        s = time.time()
        self.BVA_recover(observed_size_in_setup)
        e = time.time()
        self.time = e-s
        # print("BVARecoeryTime:{}".format(self.time))
        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        self.total_inject_length = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        self.accuracy = self.recover_queries_num/self.total_queries_num
        # print("BVARecoeryRer:{}".format(self.accuracy))
    def BVA_recover(self, observed_size_in_setup):
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
                for query_in_setup in observed_size_in_setup.keys():
                    #flag_found = False
                    if query in self.size_after_injection_padding.keys():
                        self.recover_tag[query] = (self.size_after_injection_padding[query] - observed_size_in_setup[query_in_setup]) / self.gamma
                        if self.recover_tag[query] in CW:
                            break
                """
                for group in self.recover_group[query]:
                    flag_found = False
                    for kw_id in self.Group[group]: ###############
                        if query in self.size_after_injection_padding.keys() and kw_id in observed_size_in_setup.keys(): 
                            if (self.size_after_injection_padding[query] - observed_size_in_setup[kw_id]) % self.gamma == 0:
                                self.recover_tag[query] = (self.size_after_injection_padding[query] - observed_size_in_setup[kw_id]) / self.gamma
                                flag_found = True
                                break
                    if flag_found:
                        break
                """
                
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def BVA_inject(self):
        kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
        if kws_each_doc==0:
            num_injection_doc=0
        else:
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        """
        generate injected doc
        """
        size_each_doc = []
        if num_injection_doc >= 1:
            size_each_doc.append(self.gamma)
            self.total_inject_size += size_each_doc[0]
        if num_injection_doc >= 2:
            for i in range(1, num_injection_doc):
                size_each_doc.append(size_each_doc[i-1] + size_each_doc[i-1])
                self.total_inject_size += size_each_doc[i]
        """
        size after injection
        """
        injection_length = {}
        injection_size = {}
        for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
            injection_length[kws_ind] = 0
            injection_size[kws_ind] = 0
            for num_ind in range(num_injection_doc):
                if ((kws_ind >> num_ind) & 1) == 1 :
                    injection_length[kws_ind] += 1
                    injection_size[kws_ind] += size_each_doc[num_ind]
                    #real_size_after_injection[kws_ind] += size_each_doc[num_ind]       
        self.get_size_and_length_after_injection_padding(injection_length, injection_size)

    def BVMA_SP_main(self, query_type):
        self.total_queries_num = 0
        self.recover_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0
        """
        compute frequency in baseline phase
        """
        baseline_trend_matrix = np.zeros((self.trend_matrix_norm.shape[0], len(self.observed_queries))) #行kws， 列week
        if query_type == 'real-world':
            for i_week, weekly_tags in enumerate(self.observed_queries):
                if len(weekly_tags) > 0:
                    counter = Counter(weekly_tags)
                    for key in counter:
                        baseline_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
        
        """
        Setup: get self.length_after_setup_padding, self.length_after_setup_padding
        """
        self.get_size_and_length_after_setup_padding()
        observed_size_in_setup, _, observed_length_in_setup = self.get_baseline_observed_size_and_length()
        """
        injection: get self.length_after_injection_padding, self.length_after_injection_padding
        """
        self.BVMA_SP_inject()
        """
        recovery
        """
        #observed_size_in_baseline = self.observed_size.copy()
        #observed_length_in_baseline = self.observed_length.copy()
        s = time.time()
        self.time = self.BVMA_SP_recover(query_type, baseline_trend_matrix, observed_size_in_setup, observed_length_in_setup)
        e = time.time()
        self.time = e - s
        # print("BVMARecoeryTime:{}".format(self.time))
        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        self.total_inject_length = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        self.accuracy = self.recover_queries_num/self.total_queries_num
        # print("BVMARecoeryRer:{}".format(self.accuracy))
    def BVMA_SP_recover(self, query_type, baseline_trend_matrix, observed_size_in_setup, observed_length_in_setup):
        self.real_tag = {}
        self.recover_tag = {}
        """
        compute frequency in recovery phase
        """
        if query_type == 'real-world':
            recover_trend_matrix = np.zeros((self.trend_matrix_norm.shape[0], self.trend_matrix_norm.shape[1]))
            for i_week, weekly_tags in enumerate(self.target_queries):
                if len(weekly_tags) > 0:
                    counter = Counter(weekly_tags)
                    for key in counter:
                        recover_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
            
        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                
                CW = []
                for group in self.recover_group[query]:
                    for kw_id in self.Group[group]: 
                        CW.append(kw_id)
                CA = []
                for query_in_setup in observed_size_in_setup.keys():
                    #flag_found = False
                    findflag = False
                    if query in self.size_after_injection_padding.keys() and self.size_after_injection_padding[query] - observed_size_in_setup[query_in_setup] >= 0 and (self.size_after_injection_padding[query] - observed_size_in_setup[query_in_setup]) - (len(self.chosen_kws)/2)*math.log2(len(self.chosen_kws)) < len(self.chosen_kws):    
                
                        
                #for group in self.recover_group[query]:
                #    flag_found = False
                #    for kw_id in self.Group[group]:# recover from Group
                #        if query in self.size_after_injection_padding.keys() and kw_id in observed_size_in_setup.keys() and self.size_after_injection_padding[query] - observed_size_in_setup[kw_id] >= 0 and (self.size_after_injection_padding[query] - observed_size_in_setup[kw_id]) - (len(self.chosen_kws)/2)*math.log2(len(self.chosen_kws)) < len(self.chosen_kws):                           
                        diffReBa = (int) ((self.size_after_injection_padding[query] - observed_size_in_setup[query_in_setup])/1)
                        t_counteae = 0
                        num_tF = 0
                        while(diffReBa>=0):
                            diffReBa -= (int) (len(self.chosen_kws)/2)
                            if diffReBa<0:
                                break
                            t_counteae += 1
                            if diffReBa<len(self.chosen_kws):
                                re_kw_id = diffReBa
                                tmp_kw_id = re_kw_id
                                num_tF = 0
                                while tmp_kw_id!=0:
                                    if tmp_kw_id&1==1:
                                        num_tF += 1
                                    tmp_kw_id >>= 1
                                if t_counteae!=num_tF:
                                    continue
                                if self.length_after_injection_padding[query] - observed_length_in_setup[query_in_setup] == num_tF:
                                    if re_kw_id in CW:
                                        if query_type == 'uniform':
                                            self.recover_tag[query] = re_kw_id
                                            break
                                        CA.append(re_kw_id)
                                        findflag = True
                                        break
                        if findflag:
                            break
                if query_type == 'real-world':
                    min_cost = 1000
                    real_key = -1
                    for kw in CA:
                        tmp = np.linalg.norm([[recover_trend_matrix[query][t] - baseline_trend_matrix[kw][t2] for t2 in range(baseline_trend_matrix.shape[1])] for t in range(recover_trend_matrix.shape[1])])#np.linalg.norm(
                        if tmp < min_cost:
                            min_cost = tmp
                            real_key = kw
                    self.recover_tag[query] = real_key
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def BVMA_SP_inject(self):       
        kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
        if kws_each_doc==0:
            num_injection_doc=0
        else:
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))

        size_each_doc = []
        if num_injection_doc >= 1:
            size_each_doc.append(1 + (int) (len(self.chosen_kws)/2))
            self.total_inject_size += size_each_doc[0]
            #self.total_inject_size += math.ceil(len(self.chosen_kws)/2)
        if num_injection_doc >= 2:
            for i in range(1, num_injection_doc):
                size_each_doc.append(size_each_doc[i-1] + size_each_doc[i-1] - (int) (len(self.chosen_kws)/2))
                self.total_inject_size += size_each_doc[i]
                #self.total_inject_size += math.ceil(len(self.chosen_kws)/2)
        """
        statistics
        """
        injection_length = {}
        injection_size = {}
        for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
            injection_length[kws_ind] = 0
            injection_size[kws_ind] = 0
            for num_ind in range(num_injection_doc):
                if ((kws_ind >> num_ind) & 1) == 1 :
                    injection_length[kws_ind] += 1
                    injection_size[kws_ind] += size_each_doc[num_ind]
                    #real_size_after_injection[kws_ind] += size_each_doc[num_ind]
                    #real_length_after_injection[kws_ind] += 1

        #return real_size_after_injection, real_length_after_injection
        self.injection_length_without_padding = injection_length
        self.get_size_and_length_after_injection_padding(injection_length, injection_size)

def multiprocess_worker(kw_dict, chosen_kws, docu, aa, adv_observed_offset, observed_period, target_period, number_queries_per_period, real_size, real_length, offset_of_Decoding):

    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), 260, adv_observed_offset)
    begin_time = random.randint(0, len(trend_matrix[0])-observed_period-target_period-1)

    observed_queries = utils.generate_queries(trend_matrix[:, begin_time:begin_time+observed_period], 'real-world', number_queries_per_period)
    target_queries = utils.generate_queries(trend_matrix[:, begin_time+observed_period:begin_time+observed_period+target_period], 'real-world', number_queries_per_period)


    min_file_size, max_file_size, _ = utils.get_file_size(docu)
    sdb = Attack(docu, min_file_size, max_file_size, aa, chosen_kws, observed_queries, target_queries, 1, trend_matrix, real_size, real_length, offset_of_Decoding)
    sdb.Location_query_to_group()
    Cluster_acc = sdb.cluster_acc
    # print(Cluster_acc)
    sdb.get_bandwidth()

    sdb.BVA_main((int)(len(chosen_kws)/2))
    BVA_acc = sdb.accuracy
    # print(sdb.accuracy)
    sdb.BVMA_SP_main('real-world')
    BVMA_acc = sdb.accuracy
    # print(sdb.accuracy)
    return [Cluster_acc, BVA_acc, BVMA_acc, sdb.bandwidth_setup, sdb.size_setup, sdb.bandwidth_static_query, sdb.size_static_query, 
        sdb.bandwidth_injection, sdb.size_injection, sdb.bandwidth_dynamic_query, sdb.size_dynamic_query]

def plot_figure_location(Cluster_acc):

    #BVA_acc = [[0.3215, 0.3932, 0.354, 0.3461, 0.3691, 0.3598, 0.3262, 0.3166, 0.3232, 0.2956, 0.3735, 0.3166, 0.2756, 0.327, 0.356, 0.357, 0.2781, 0.3304, 0.3447, 0.3833, 0.3264, 0.2764, 0.3589, 0.3479, 0.3173, 0.3375, 0.3477, 0.326, 0.3057, 0.2773], [0.1836, 0.1439, 0.1364, 0.1577, 0.1532, 0.1803, 0.1474, 0.1483, 0.1607, 0.1678, 0.1211, 0.1601, 0.1835, 0.207, 0.1516, 0.1595, 0.19, 0.1686, 0.1472, 0.1769, 0.2289, 0.1729, 0.208, 0.1282, 0.191, 0.1646, 0.1789, 0.1516, 0.1404, 0.1629], [0.0645, 0.1126, 0.1207, 0.0644, 0.1148, 0.0597, 0.067, 0.0804, 0.1556, 0.064, 0.0759, 0.0642, 0.0808, 0.0788, 0.0613, 0.0787, 0.0625, 0.0707, 0.0662, 0.0589, 0.0616, 0.0955, 0.0687, 0.0677, 0.0865, 0.0463, 0.0581, 0.0609, 0.0829, 0.0654], [0.0196, 0.0326, 0.0274, 0.033, 0.0271, 0.0196, 0.0223, 0.0231, 0.0204, 0.0277, 0.0278, 0.0253, 0.0341, 0.0259, 0.0381, 0.0264, 0.022, 0.0194, 0.0295, 0.0863, 0.0306, 0.0197, 0.0269, 0.0347, 0.0296, 0.0273, 0.0169, 0.0154, 0.0847, 0.0209]]

    #BVMA_acc = [[0.3215, 0.3928, 0.354, 0.3461, 0.3691, 0.3598, 0.3262, 0.3154, 0.3227, 0.295, 0.3731, 0.3164, 0.2754, 0.327, 0.3559, 0.3567, 0.2781, 0.3304, 0.3445, 0.3802, 0.3264, 0.2763, 0.3589, 0.3479, 0.3173, 0.3375, 0.3477, 0.3255, 0.3053, 0.2773], [0.1836, 0.1446, 0.1355, 0.1578, 0.1532, 0.18, 0.1471, 0.1483, 0.1607, 0.1673, 0.1202, 0.16, 0.1807, 0.2071, 0.1513, 0.1551, 0.1915, 0.1685, 0.1467, 0.1763, 0.2283, 0.1725, 0.2075, 0.1271, 0.1914, 0.1646, 0.1788, 0.1496, 0.1404, 0.1601], [0.063, 0.1121, 0.1194, 0.0639, 0.1149, 0.0597, 0.0662, 0.0804, 0.1551, 0.0607, 0.0756, 0.064, 0.0813, 0.0792, 0.06, 0.0786, 0.0639, 0.0699, 0.0658, 0.0585, 0.0616, 0.0941, 0.0682, 0.0674, 0.0866, 0.0458, 0.0576, 0.0607, 0.0785, 0.0651], [0.0184, 0.0326, 0.0256, 0.033, 0.0275, 0.0183, 0.0227, 0.0231, 0.0201, 0.0233, 0.0293, 0.0259, 0.0313, 0.025, 0.0382, 0.0266, 0.0218, 0.0212, 0.0286, 0.0876, 0.025, 0.0203, 0.0251, 0.0336, 0.0286, 0.0241, 0.0167, 0.0171, 0.0845, 0.0189]]

    labels = [r'$\alpha$=2', r'$\alpha$=8', r'$\alpha$=32', r'$\alpha$=128']
    c = []
    for i in range(len(Cluster_acc)):
        for j in range(len(Cluster_acc[0])):
            c.append(['ClusterLoc', labels[i], Cluster_acc[i][j]])

    plt.clf()
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    #"text.usetex": True,
    #"font.family": "serif",
    "font.serif": ["Times"], #注意这里是Times，不是Times New Roman
    "font.size":30,
    "lines.markersize":20})

    fig, ax = plt.subplots()

    plt.plot(labels, np.mean(Cluster_acc, axis=1), label = 'Accuracy', color="lightgreen",markeredgecolor='green',marker="o") # '--g'
    plt.ylabel("Location Accuracy")
    plt.legend()
    plt.savefig(utils.PLOTS_PATH + '/' + 'ShieldDBEnronClusterLocation.pdf', bbox_inches = 'tight', dpi = 600)
    plt.show()

def plot_figure_rerAndoverhead(BVA_acc, BVMA_acc, Bandwidth_setup, Bandwidth_static_query,
        Bandwidth_update, Bandwidth_dynamic_query):

    #BVA_acc = [[0.3215, 0.3932, 0.354, 0.3461, 0.3691, 0.3598, 0.3262, 0.3166, 0.3232, 0.2956, 0.3735, 0.3166, 0.2756, 0.327, 0.356, 0.357, 0.2781, 0.3304, 0.3447, 0.3833, 0.3264, 0.2764, 0.3589, 0.3479, 0.3173, 0.3375, 0.3477, 0.326, 0.3057, 0.2773], [0.1836, 0.1439, 0.1364, 0.1577, 0.1532, 0.1803, 0.1474, 0.1483, 0.1607, 0.1678, 0.1211, 0.1601, 0.1835, 0.207, 0.1516, 0.1595, 0.19, 0.1686, 0.1472, 0.1769, 0.2289, 0.1729, 0.208, 0.1282, 0.191, 0.1646, 0.1789, 0.1516, 0.1404, 0.1629], [0.0645, 0.1126, 0.1207, 0.0644, 0.1148, 0.0597, 0.067, 0.0804, 0.1556, 0.064, 0.0759, 0.0642, 0.0808, 0.0788, 0.0613, 0.0787, 0.0625, 0.0707, 0.0662, 0.0589, 0.0616, 0.0955, 0.0687, 0.0677, 0.0865, 0.0463, 0.0581, 0.0609, 0.0829, 0.0654], [0.0196, 0.0326, 0.0274, 0.033, 0.0271, 0.0196, 0.0223, 0.0231, 0.0204, 0.0277, 0.0278, 0.0253, 0.0341, 0.0259, 0.0381, 0.0264, 0.022, 0.0194, 0.0295, 0.0863, 0.0306, 0.0197, 0.0269, 0.0347, 0.0296, 0.0273, 0.0169, 0.0154, 0.0847, 0.0209]]

    #BVMA_acc = [[0.3215, 0.3928, 0.354, 0.3461, 0.3691, 0.3598, 0.3262, 0.3154, 0.3227, 0.295, 0.3731, 0.3164, 0.2754, 0.327, 0.3559, 0.3567, 0.2781, 0.3304, 0.3445, 0.3802, 0.3264, 0.2763, 0.3589, 0.3479, 0.3173, 0.3375, 0.3477, 0.3255, 0.3053, 0.2773], [0.1836, 0.1446, 0.1355, 0.1578, 0.1532, 0.18, 0.1471, 0.1483, 0.1607, 0.1673, 0.1202, 0.16, 0.1807, 0.2071, 0.1513, 0.1551, 0.1915, 0.1685, 0.1467, 0.1763, 0.2283, 0.1725, 0.2075, 0.1271, 0.1914, 0.1646, 0.1788, 0.1496, 0.1404, 0.1601], [0.063, 0.1121, 0.1194, 0.0639, 0.1149, 0.0597, 0.0662, 0.0804, 0.1551, 0.0607, 0.0756, 0.064, 0.0813, 0.0792, 0.06, 0.0786, 0.0639, 0.0699, 0.0658, 0.0585, 0.0616, 0.0941, 0.0682, 0.0674, 0.0866, 0.0458, 0.0576, 0.0607, 0.0785, 0.0651], [0.0184, 0.0326, 0.0256, 0.033, 0.0275, 0.0183, 0.0227, 0.0231, 0.0201, 0.0233, 0.0293, 0.0259, 0.0313, 0.025, 0.0382, 0.0266, 0.0218, 0.0212, 0.0286, 0.0876, 0.025, 0.0203, 0.0251, 0.0336, 0.0286, 0.0241, 0.0167, 0.0171, 0.0845, 0.0189]]

    labels = [r'$\alpha$=2', r'$\alpha$=8', r'$\alpha$=32', r'$\alpha$=128']
    c = []
    for i in range(len(BVA_acc)):
        for j in range(len(BVA_acc[0])):
            #c.append(['ClusterLoc', labels[i], Cluster_acc[i][j]])
            c.append(['BVA', labels[i], BVA_acc[i][j]])
            c.append(['BVMA', labels[i], BVMA_acc[i][j]])


    bandwidth_setup = np.mean(Bandwidth_setup, axis=1)
    bandwidth_static_query = np.mean(Bandwidth_static_query, axis=1)
    bandwidth_update = np.mean(Bandwidth_update, axis=1)
    bandwidth_dynamic_query = np.mean(Bandwidth_dynamic_query, axis=1)
    df = pd.DataFrame(c, columns=['Attack-Rer', '', 'Recovery rate']) 
    print(df)

    band = []
    for i in range(len(bandwidth_setup)):
        band.append([r'Setup$\&$Fill', labels[i], bandwidth_setup[i]*100])
        band.append(['S-Query', labels[i], bandwidth_static_query[i]*100])
        band.append([r'Inj$\&$Fill', labels[i], bandwidth_update[i]*100])
        band.append(['I-Query', labels[i], bandwidth_dynamic_query[i]*100])

    df2 = pd.DataFrame(band, columns=['Phase-Bandwidth', '', r'Bandwidth Overhead ($\%$)']) 

    fig, ax = plt.subplots()


    #XX_BVA = [0, 1, 2, 3]
    #XX_Cluster = [i-0.1 for i in XX_BVA]
    #XX_BVMA = [i+0.1 for i in XX_BVA]
    #ax2=ax.twinx()
    #plt.plot(XX_BVA, np.mean(BVA_acc, axis=1), '--b')
    #plt.plot(XX_Cluster, np.mean(Cluster_acc, axis=1), '--g')
    #plt.plot(XX_BVMA, np.mean(BVMA_acc, axis=1), '--r')

    plt.clf()
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], #注意这里是Times，不是Times New Roman
    "font.size":30,
    "lines.markersize":20})

    #XX_BVA = [0, 1, 2, 30]
    #XX_BVMA = [i+0.1 for i in XX_BVA]
    #ax2=ax.twinx()
    #plt.plot(XX_BVA, np.mean(BVA_acc, axis=1), '--b')
    #plt.plot(XX_BVMA, np.mean(BVMA_acc, axis=1), '--r')

    df2 = pd.DataFrame(band, columns=['Phase-Overhead', '', r'Overhead ($\%$)']) 

    fig, ax = plt.subplots()
    pale = {"BVA": 'skyblue', "BVMA": 'red'}
    sns.boxplot(x = '', y = 'Recovery rate', hue = 'Attack-Rer',data=df, palette=pale, width=0.25,linewidth=1)
    ax.legend(loc = 2, bbox_to_anchor = (1.2, 1.1))
    #ax.grid(which='both', axis='x')
    #plt.vlines(0.3, -1, 1) U-Query
    ax2 = ax.twinx()
    markers = {r'Setup$\&$Fill': 'X', 'S-Query': '<', r'Inj$\&$Fill': 'o', 'I-Query': '>'}
    #sns.pairplot(markers = ['o', 'x', '^'], hue = 'Phase-Bandwidth', palette='deep', data=df2)
    sns.scatterplot(x='', y=r'Overhead ($\%$)', style = 'Phase-Overhead', hue = 'Phase-Overhead', markers=markers,  ax=ax2, data=df2)
    XX = [0, 1, 2, 3]
    ax2.plot(XX, [i*100 for i in bandwidth_setup],  '--b')
    ax2.plot(XX, [i*100 for i in bandwidth_static_query], color='orange',ls='--')
    ax2.plot(XX, [i*100 for i in bandwidth_update], '--g')
    ax2.plot(XX, [i*100 for i in bandwidth_dynamic_query], color='salmon',ls='--')
    #ax2.plot(xvalues, [(oh-1)*100 for oh in bwvals], 'rx') style = 'Phase-Bandwidth', palette='deep', hue = 'Phase-Bandwidth',  
    #ax2.legend(loc=1) color = ['r', 'g', 'b'], s = 120, ax.legend(loc = 2, bbox_to_anchor = (1.2, 1.1))
    ax2.legend(loc = 2, bbox_to_anchor = (1.2, 0.7))
    plt.savefig(utils.PLOTS_PATH + '/' + 'ShieldDBEnronLocation.pdf', bbox_inches = 'tight', dpi = 600)

    plt.show()

if __name__=='__main__': 
   

    with open(os.path.join(utils.RESULT_PATH, 'ShieldDBLocationEnron.pkl'), 'rb') as f:
        (Cluster_acc, BVA_acc, BVMA_acc, Bandwidth_setup, Size_setup, Bandwidth_static_query, Size_static_query, Bandwidth_injection, Size_injection, Bandwidth_dynamic_query, Size_dynamic_query) = pickle.load(f)
        f.close()
    print(Cluster_acc)
    print(BVA_acc)
    print(BVMA_acc)
    print(Bandwidth_setup)
    print(Size_setup)
    print(Bandwidth_static_query)
    print(Size_static_query)
    print(Bandwidth_injection)
    print(Size_injection)
    print(Bandwidth_dynamic_query)
    print(Size_dynamic_query)

    #plot_figure_location(Cluster_acc)

    Cluster_acc = []
    BVA_acc = []
    BVMA_acc = []
    Bandwidth_setup = []
    Size_setup = []
    Bandwidth_static_query = []
    Size_static_query = []
    Bandwidth_injection = []
    Size_injection = []
    Bandwidth_dynamic_query = []
    Size_dynamic_query = []
    ShieldDBAlpha = [2, 8, 32, 128] # 
    pbar = tqdm(total=len(ShieldDBAlpha))
    loop = 0
    for aa in ShieldDBAlpha:
        Cluster_tmp_acc = []
        BVA_tmp_acc = []
        BVMA_tmp_acc = []
        Bandwidth_tmp_setup = []
        Bandwidth_tmp_static_query = []
        Bandwidth_tmp_injection = []
        Bandwidth_tmp_dynamic_query = []
        Size_tmp_setup = []
        Size_tmp_static_query = []
        Size_tmp_injection = []
        Size_tmp_dynamic_query = []
        partial_function = partial(multiprocess_worker, kw_dict, chosen_kws, docu, aa, adv_observed_offset, observed_period, target_period, number_queries_per_period, real_size, real_length)
        with Pool(processes=exp_times) as pool:
            for result in pool.map(partial_function, offset_of_Decoding_list):
                Cluster_tmp_acc.append(result[0])
                BVA_tmp_acc.append(result[1])
                BVMA_tmp_acc.append(result[2])
                Bandwidth_tmp_setup.append(result[3])
                Size_tmp_setup.append(result[4])
                Bandwidth_tmp_static_query.append(result[5])
                Size_tmp_static_query.append(result[6])
                Bandwidth_tmp_injection.append(result[7])
                Size_tmp_injection.append(result[8])
                Bandwidth_tmp_dynamic_query.append(result[9])
                Size_tmp_dynamic_query.append(result[10])

        Cluster_acc.append(Cluster_tmp_acc)
        BVA_acc.append(BVA_tmp_acc)
        BVMA_acc.append(BVMA_tmp_acc)
        Bandwidth_setup.append(Bandwidth_tmp_setup)
        Size_setup.append(Size_tmp_setup)
        Bandwidth_static_query.append(Bandwidth_tmp_static_query)
        Size_static_query.append(Size_tmp_static_query)
        Bandwidth_injection.append(Bandwidth_tmp_injection)
        Size_injection.append(Size_tmp_injection)
        Bandwidth_dynamic_query.append(Bandwidth_tmp_dynamic_query)
        Size_dynamic_query.append(Size_tmp_dynamic_query)
        pbar.update(math.ceil((loop+1)/len(ShieldDBAlpha)))
        loop += 1
    pbar.close()
    with open(os.path.join(utils.RESULT_PATH, 'ShieldDBLocationEnron.pkl'), 'wb') as f:
        pickle.dump((Cluster_acc, BVA_acc, BVMA_acc, Bandwidth_setup, Size_setup, Bandwidth_static_query, Size_static_query, Bandwidth_injection, Size_injection, Bandwidth_dynamic_query, Size_dynamic_query), f)
        f.close()
    print(Cluster_acc)
    print(BVA_acc)
    print(BVMA_acc)
    print(Bandwidth_setup)
    print(Size_setup)
    print(Bandwidth_static_query)
    print(Size_static_query)
    print(Bandwidth_injection)
    print(Size_injection)
    print(Bandwidth_dynamic_query)
    print(Size_dynamic_query)

    plot_figure_location(Cluster_acc)
    plot_figure_rerAndoverhead(BVA_acc, BVMA_acc, Bandwidth_setup, Bandwidth_static_query,
        Bandwidth_injection, Bandwidth_dynamic_query)
   