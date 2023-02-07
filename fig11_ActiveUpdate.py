import time
import math
import numpy as np
from collections import Counter

from torch import rand
import utils 
import os
import pickle
import random
from multiprocessing import Pool
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
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
EXP_TIM = 20
offset_of_Decoding_list = [offset_of_Decoding]*EXP_TIM




class Attack: 
    def __init__(self, kw_to_id, real_doc, sample_doc, min_file_size, max_file_size, update_percentage_with_injectoion, chosen_kws, observed_queries, target_queries, kws_leak_percent, trend_matrix_norm, real_size, real_length, offset_of_Decoding):
        self.real_tag = {}
        self.recover_tag = {}

        self.setup_bandwidth = 0
        self.update_bandwidth = 0
        self.query_bandwidth = 0

        self.kw_to_id = kw_to_id
        #print(len(self.kw_to_id))
        self.real_doc = real_doc
        self.sample_doc = sample_doc

        self.time = 0
        self.inject_time = 0        
        self.recover_queries_num = 0
        self.total_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0

        self.injection_length = 0

        self.bandwidth_setup = 0 #setup database
        self.bandwidth_update = 0 #a file contains multiple keywords, BVA, BVMA ;no padding of Decoding, impractical of SR.
        self.bandwidth_query = 0 #return No. of files

        self.kws_leak_percent = kws_leak_percent
        self.chosen_kws = chosen_kws
        self.target_queries = target_queries
        self.observed_queries = observed_queries
        self.trend_matrix_norm = trend_matrix_norm
        
        self.update_percentage_with_injectoion = update_percentage_with_injectoion
  
        self.size_at_setup, self.length_at_setup = real_size, real_length 
        self.size_without_update, self.length_without_update = real_size, real_length 
        #self.size_with_update, self.length_with_update = real_size, real_length 
        #self.size_after_setup_padding, self.length_after_setup_padding = {}, {}
        #self.size_after_injection_padding, self.length_after_injection_padding = {}, {}
        self.min_file_size, self.max_file_size = min_file_size, max_file_size
        self.injection_length_without_padding = {} 
        self.length_after_injection_and_update, self.size_after_injection_and_update = {},{}
        self.size_after_injection_and_update = {}
        """
        baseline phase
        """
        #self.observed_size, self.max_observed_size, self.observed_length = self.get_baseline_observed_size_and_length(real_size, real_length)
        """
        get offset of Decoding
        """
        self.offset = offset_of_Decoding
        #self.Group = self.Group_cluster()
    def get_size_and_length_after_injection_and_update(self, injection_length, injection_size, update_length, update_size):
        self.length_after_injection_and_update = {}
        self.size_after_injection_and_update = {}
        for k in self.size_at_setup.keys():
            self.length_after_injection_and_update[k] = self.length_at_setup[k]
            self.size_after_injection_and_update[k] = self.size_at_setup[k]
            if k in injection_length.keys():
                self.length_after_injection_and_update[k] += injection_length[k]
                self.size_after_injection_and_update[k] += injection_size[k]
            if k in update_length.keys():
                self.length_after_injection_and_update[k] += update_length[k]
                self.size_after_injection_and_update[k] += update_size[k]


    def random_update_database(self): #kw_to_id, real_length, real_size, real_doc, sample_doc
        operation_type = ['add', 'delete']
        update_length = {}
        update_size = {}
        # print((int) (self.update_percentage_with_injectoion*self.injection_length))
        for _ in range((int) (self.update_percentage_with_injectoion*self.injection_length)):         
            op = random.choice(operation_type)
            if len(self.sample_doc)==0 and len(self.real_doc)==0:
                break
            #if len(self.sample_doc)==0 and len(self.real_doc)==0: # random generate doc. and kws.
            #    operation = 'add'
            #    kws_list = random.sample(range(len(chosen_kws)), random.choice(range(1, len(chosen_kws)))) # chose random kws as add.
            #    doc_size = random.choice(range(min(len(kws_list), self.min_file_size), max(len(kws_list), self.max_file_size)))
            #    for kw_id in kws_list:
            #        update_length[kw_id] += 1
            #        update_size[kw_id] += doc_size
            if op=='add':
                if len(self.sample_doc)==0:
                    continue
                add_doc_id = random.choice(range(len(self.sample_doc)))
                add_doc = list(set(self.sample_doc[add_doc_id]))
                for kw in add_doc:
                    if kw not in self.kw_to_id.keys():
                        continue
                    if self.kw_to_id[kw] in update_length.keys():
                        update_length[self.kw_to_id[kw]] += 1
                        update_size[self.kw_to_id[kw]] += len(add_doc)
                    else:
                        update_length[self.kw_to_id[kw]] = 1
                        update_size[self.kw_to_id[kw]] = len(add_doc)
                self.real_doc.append(self.sample_doc.pop(add_doc_id))
            else:
                if len(self.real_doc)==0:
                    continue
                delete_doc_id = random.choice(range(len(self.real_doc)))
                #print(self.real_doc[delete_doc_id])
                delete_doc = list(set(self.real_doc[delete_doc_id]))
                for kw in delete_doc:
                    if kw not in self.kw_to_id.keys():
                        continue
                    if self.kw_to_id[kw] in update_length.keys():
                        update_length[self.kw_to_id[kw]] -= 1
                        update_size[self.kw_to_id[kw]] -= len(delete_doc)
                    else:
                        update_length[self.kw_to_id[kw]] = -1
                        update_size[self.kw_to_id[kw]] = -len(delete_doc)        
                self.real_doc.pop(delete_doc_id)
        return update_length, update_size

    def get_baseline_observed_size_and_length(self):
        """
        observe size and length in baseline phase
        """
        observed_size = {}
        observed_length = {}
        max_observed_size = 0
        for i_week in self.observed_queries:
            for query in i_week:
                if query not in self.size_at_setup.keys():
                    observed_size[query] = 0
                    observed_length[query] = 0
                else:
                    observed_size[query] = self.size_at_setup[query]
                    observed_length[query] = self.length_at_setup[query]
                    if max_observed_size < observed_size[query]:
                        max_observed_size = observed_size[query]
        return observed_size, max_observed_size, observed_length

    def Decoding_main(self):
        self.total_queries_num = 0
        self.recover_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0
        """
        Setup: get self.length_after_setup_padding, self.length_after_setup_padding
        """
        observed_size_in_setup, _, _ = self.get_baseline_observed_size_and_length()
        """
        injection: get self.length_after_injection_padding, self.length_after_injection_padding
        """
        self.Decoding_inject()
        """
        recovery
        """
        #observed_size_in_baseline = self.observed_size.copy()
        s = time.time()
        self.Decoding_recover(observed_size_in_setup)
        e = time.time()
        self.time = e - s
        # print("DecodingRecoeryTime:{}".format(self.time))
        
        self.total_inject_length = (int) (len(self.chosen_kws)*self.kws_leak_percent) - 1
        self.accuracy = self.recover_queries_num/self.total_queries_num
    def Decoding_recover(self, observed_size_in_setup):
        self.real_tag = {}
        self.recover_tag = {}
        
        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                for kw_id in observed_size_in_setup.keys():
                    if query in self.size_after_injection_and_update.keys(): 
                        if (self.size_after_injection_and_update[query] - observed_size_in_setup[kw_id]) % self.offset == 0:
                            self.recover_tag[query] = (self.size_after_injection_and_update[query] - observed_size_in_setup[kw_id]) / self.offset
                            break
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def Decoding_inject(self):
        """
        injection: injection size and real_size_after_injection
        """
        injection_length = {}
        injection_size = {}
        for kw_id in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
            injection_length[kw_id] = 1
            injection_size[kw_id] = kw_id*self.offset
            #self.total_inject_size += kw_id*self.offset 
            #real_size_after_injection[kw_id] += kw_id*self.offset
        self.injection_length = len(injection_length)
        update_length, update_size = self.random_update_database()
        self.get_size_and_length_after_injection_and_update(injection_length, injection_size, update_length, update_size)
         
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
        #self.get_size_and_length_after_setup_padding()
        observed_size_in_setup, _, _ = self.get_baseline_observed_size_and_length()
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
    def BVA_recover(self, observed_size_in_setup):
        self.real_tag = {}
        self.recover_tag = {}

        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                for kw_id in observed_size_in_setup.keys():
                    if query in self.size_after_injection_and_update.keys(): 
                        if (self.size_after_injection_and_update[query] - observed_size_in_setup[kw_id]) % self.gamma == 0:
                            self.recover_tag[query] = (self.size_after_injection_and_update[query] - observed_size_in_setup[kw_id]) / self.gamma
                            break
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def BVA_inject(self):
        kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
        if kws_each_doc==0:
            num_injection_doc=0
        else:
            num_injection_doc = math.ceil(np.log2(len(self.chosen_kws)))

        self.injection_length = num_injection_doc
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
        update_length, update_size = self.random_update_database()
        self.get_size_and_length_after_injection_and_update(injection_length, injection_size, update_length, update_size)
                    #real_size_after_injection[kws_ind] += size_each_doc[num_ind]       
        #self.get_size_and_length_after_injection(injection_length, injection_size)

    def SR_main(self, m):
        self.total_queries_num = 0
        self.recover_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0
        self.m = m
        """
        Setup: get self.length_after_setup_padding, self.length_after_setup_padding
        """
        #self.get_size_and_length_after_setup_padding()
        _, _, observed_length_in_setup = self.get_baseline_observed_size_and_length()
        """
        injection: get self.length_after_injection_padding, self.length_after_injection_padding
        """
        self.SR_inject()
        """
        recovery
        """
        s = time.time()
        self.SR_recover(observed_length_in_setup)
        e = time.time()
        self.time = e-s
        # print("SR-{}RecoeryTime:{}".format(self.m, self.time))
        #print("SR-Acc:{}".format(self.accuracy))

        self.accuracy = self.recover_queries_num/self.total_queries_num
        # print("SR-Acc:{}".format(self.accuracy))
    def SR_recover(self, observed_length_in_setup):
        self.real_tag = {}
        self.recover_tag = {}
        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                for kw_id in observed_length_in_setup.keys():
                    if query in self.length_after_injection_and_update.keys(): 
                        if (self.length_after_injection_and_update[query] - observed_length_in_setup[kw_id]) % self.m == 0:
                            self.recover_tag[query] = (self.length_after_injection_and_update[query] - observed_length_in_setup[kw_id]) / self.m - 1
                            break
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def SR_inject(self):
        """
        generate doc
        """    
        #size_each_doc = []
        #for i in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
        #    doc_contain_ith_kws = []
        #    for j in range(self.m):
        #        doc_contain_ith_kws.append(i+1)
        #    size_each_doc.append(doc_contain_ith_kws)

        known_kws_num = (int) (len(self.chosen_kws)*self.kws_leak_percent)
        self.total_inject_length = self.m*known_kws_num
        self.injection_length = self.total_inject_length
        self.total_inject_size = self.m*self.kws_leak_percent*known_kws_num*(known_kws_num+1)/2
        #self.injection_length
        injection_length = {}
        injection_size = {}
        for kws_ind in range(known_kws_num):
            injection_length[kws_ind] = (kws_ind+1)*self.m
            injection_size[kws_ind] = (kws_ind+1)*self.m # non-necessary
            #real_length_after_injection[kws_ind] += (kws_ind+1)*self.m     
        update_length, update_size = self.random_update_database()
        self.get_size_and_length_after_injection_and_update(injection_length, injection_size, update_length, update_size)
              
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
        #self.get_size_and_length_after_setup_padding()
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
        # self.size_after_injection_and_update
        # self.length_after_injection_and_update    
        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                CA = []
                findflag = False
                for kw_id in observed_size_in_setup.keys():
                    if query in self.size_after_injection_and_update.keys() and self.size_after_injection_and_update[query] - observed_size_in_setup[kw_id] >= 0 and (self.size_after_injection_and_update[query] - observed_size_in_setup[kw_id]) - (len(self.chosen_kws)/2)*math.log2(len(self.chosen_kws)) < len(self.chosen_kws):                           
                        diffReBa = (int) ((self.size_after_injection_and_update[query] - observed_size_in_setup[kw_id])/1)
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
                                if self.length_after_injection_and_update[query] - observed_length_in_setup[kw_id] == num_tF:
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
            num_injection_doc = math.ceil(np.log2(len(self.chosen_kws)))
        self.injection_length = num_injection_doc

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
        update_length, update_size = self.random_update_database()
        self.get_size_and_length_after_injection_and_update(injection_length, injection_size, update_length, update_size)

def multiprocess_worker(kw_dict, chosen_kws, docu, update_percentage_with_injectoion, adv_observed_offset, observed_period, target_period, number_queries_per_period, kw_to_id, offset_of_Decoding):

    random.shuffle(docu)
    real_doc = docu[:(int)(len(docu)/2)]
    sample_doc = docu[(int)(len(docu)/2):]
    
    real_size, real_length = utils.get_kws_size_and_length(real_doc, chosen_kws)
    min_file_size, max_file_size,_ = utils.get_file_size(real_doc)
    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), 260, adv_observed_offset)
    begin_time = random.randint(0, len(trend_matrix[0])-observed_period-target_period-1)
   
    observed_queries = utils.generate_queries(trend_matrix[:, begin_time:begin_time+observed_period], 'real-world', number_queries_per_period)
    target_queries = utils.generate_queries(trend_matrix[:, begin_time+observed_period:begin_time+observed_period+target_period], 'real-world', number_queries_per_period)
    kws_leak_percent = 1
    UPDATE = Attack(kw_to_id, real_doc, sample_doc, min_file_size, max_file_size, update_percentage_with_injectoion, chosen_kws, observed_queries, target_queries, kws_leak_percent, trend_matrix, real_size, real_length, offset_of_Decoding)
    UPDATE.BVA_main((int)(len(chosen_kws)/2))
    BVA_acc = UPDATE.accuracy

    UPDATE.BVMA_SP_main('real-world')
    BVMA_acc = UPDATE.accuracy
    UPDATE.Decoding_main()
    Decoding_acc = UPDATE.accuracy

    UPDATE.SR_main(len(chosen_kws))
    SR_W_acc= UPDATE.accuracy
    return [BVA_acc, BVMA_acc, Decoding_acc]
def plot_figure(BVA_acc, BVMA_acc, Decoding_acc):
    
    #fig = plt.figure()  # 创建画布
    #ax = plt.subplot()  # 创建作图区域
    UP_P = [0, 0.1, 0.2, 0.5, 1, 2, 5]
    UP_PER = [i*100 for i in UP_P]
    labels = UP_PER
    c = []
    for i in range(len(BVA_acc)):
        for j in range(len(BVA_acc[0])):
            c.append(['BVA', labels[i], BVA_acc[i][j]])
            c.append(['BVMA', labels[i], BVMA_acc[i][j]])
            c.append(['Decoding', labels[i], Decoding_acc[i][j]])


    df = pd.DataFrame(c, columns=['', r'Update Percentage ($\#$Upd/$\#$Inj, $\%$)', 'Recovery rate'])  #Attack
    print(df)

    plt.clf()
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], #注意这里是Times，不是Times New Roman
    "font.size":15,
    "lines.markersize":10})

    pale = {"BVA": 'lightgreen', "BVMA": 'salmon', "Decoding": 'skyblue'}
    #
    XX_BVMA = [0, 1, 2, 3, 4, 5, 6]
    XX_BVA = [i-0.2 for i in XX_BVMA]
    XX_Decoding = [i+0.2 for i in XX_BVMA]
    
    plt.plot(XX_BVA, np.mean(BVA_acc, axis=1), '--g')
    plt.plot(XX_BVMA, np.mean(BVMA_acc, axis=1), '--r') 
    plt.plot(XX_Decoding, np.mean(Decoding_acc, axis=1), '--b') 

    sns.boxplot(x = r'Update Percentage ($\#$Upd/$\#$Inj, $\%$)', y = 'Recovery rate', hue = '',data=df, palette=pale,width=0.5,linewidth=0.7)
    plt.ylim(0, 1)
    plt.vlines(0.5, 0, 1, colors='silver')
    plt.vlines(1.5, 0, 1, colors='silver')
    plt.vlines(2.5, 0, 1, colors='silver')
    plt.vlines(3.5, 0, 1, colors='silver')
    plt.vlines(4.5, 0, 1, colors='silver')
    plt.vlines(5.5, 0, 1, colors='silver')
    plt.savefig(utils.PLOTS_PATH + '/' + 'UpdateEnron.pdf', bbox_inches = 'tight', dpi = 600)
    plt.show()

if __name__=='__main__':
    kw_to_id = utils.get_kws_id(chosen_kws)
    # print(len(chosen_kws))
    UP_PER = [0, 0.1, 0.2, 0.5, 1, 2, 5] #, 10, 50, 100, 1000
    
    Up_num_BVA = []
    BVA_acc = []
    BVMA_acc = []
    Decoding_acc = []
    exp_times = EXP_TIM
    pbar = tqdm(total=len(UP_PER))
    loop = 0
    for update_percentage_with_injectoion in UP_PER:
        BVA_tmp_acc = []
        BVMA_tmp_acc = []
        Decoding_tmp_acc = []
        partial_function = partial(multiprocess_worker, kw_dict, chosen_kws, docu, update_percentage_with_injectoion, adv_observed_offset, observed_period, target_period, number_queries_per_period, kw_to_id)
        with Pool(processes=exp_times) as pool:
            for result in pool.map(partial_function, offset_of_Decoding_list):
                BVA_tmp_acc.append(result[0])
                BVMA_tmp_acc.append(result[1])
                Decoding_tmp_acc.append(result[2])


        BVA_acc.append(BVA_tmp_acc)
        BVMA_acc.append(BVMA_tmp_acc)
        Decoding_acc.append(Decoding_tmp_acc)
        pbar.update(math.ceil((loop+1)/len(UP_PER)))
        loop += 1
    pbar.close()

    print(BVA_acc)
    print(BVMA_acc)
    print(Decoding_acc)
    with open(os.path.join(utils.RESULT_PATH, 'ActiveUpdateRer.pkl'), 'wb') as f:
        pickle.dump((BVA_acc, BVMA_acc, Decoding_acc), f)
        f.close()
    plot_figure(BVA_acc, BVMA_acc, Decoding_acc)
    

