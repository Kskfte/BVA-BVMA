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
EXP_TIM = 5


class Attack: 
    def __init__(self, kw_to_id, real_doc, sample_doc, min_file_size, max_file_size, update_percentage_with_injectoion, chosen_kws, observed_queries, target_queries, kws_leak_percent, trend_matrix_norm, real_size, real_length, offset_of_Decoding, up_dis):
        self.real_tag = {}
        self.recover_tag = {}

        self.up_dis = up_dis
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

        """
        cousnt = 0
        for k in injection_length.keys():
            if k in update_size.keys():
                if self.size_at_setup[k]+update_size[k]<self.gamma:
                    count += 1
            else:
                if self.size_at_setup[k]<self.gamma:
                    count+=1
        print(count)
        """



    def random_update_database(self): #kw_to_id, real_length, real_size, real_doc, sample_doc
        operation_type = ['add', 'delete']
        update_length = {}
        update_size = {}
        # print((int) (self.update_percentage_with_injectoion*self.injection_length))
        # print(self.update_percentage_with_injectoion*len(self.real_doc))
        update_count = (int) (self.update_percentage_with_injectoion*len(self.real_doc))
        #print(update_count)
        for _ in range(update_count):    
            if self.up_dis=='AllAdd':
                op = 'add'
            elif self.up_dis=='Uniform':
                op = random.choice(operation_type)
            else:
                op = 'delete'     
            if len(self.sample_doc)==0 and len(self.real_doc)==0:
                break
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
                        update_size[self.kw_to_id[kw]] += len(self.sample_doc[add_doc_id])
                    else:
                        update_length[self.kw_to_id[kw]] = 1
                        update_size[self.kw_to_id[kw]] = len(self.sample_doc[add_doc_id])
                self.real_doc.append(self.sample_doc.pop(add_doc_id))
            else:
                if len(self.real_doc)==0:
                    continue
                delete_doc_id = random.choice(range(len(self.real_doc)))
                delete_doc = list(set(self.real_doc[delete_doc_id]))
                for kw in delete_doc:
                    if kw not in self.kw_to_id.keys():
                        continue
                    if self.kw_to_id[kw] in update_length.keys():
                        update_length[self.kw_to_id[kw]] -= 1
                        update_size[self.kw_to_id[kw]] -= len(self.real_doc[delete_doc_id])
                    else:
                        update_length[self.kw_to_id[kw]] = -1
                        update_size[self.kw_to_id[kw]] = -len(self.real_doc[delete_doc_id])      
                self.real_doc.pop(delete_doc_id)
        #print(len(self.real_doc))
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
        self.BVA_recover()
        e = time.time()
        self.time = e-s
        # print("BVARecoeryTime:{}".format(self.time))
        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        self.total_inject_length = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        self.accuracy = self.recover_queries_num/self.total_queries_num
    def BVA_recover(self):
        self.real_tag = {}
        self.recover_tag = {}

        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = math.floor(self.size_after_injection_and_update[query]/self.gamma)
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

def process(kw_dict, chosen_kws, docu, adv_observed_offset, observed_period, target_period, number_queries_per_period, kw_to_id, update_dis, Gama_Update):

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
    UPDATE = Attack(kw_to_id, real_doc, sample_doc, min_file_size, max_file_size, Gama_Update[1], chosen_kws, observed_queries, target_queries, kws_leak_percent, trend_matrix, real_size, real_length, offset_of_Decoding, update_dis)

    UPDATE.BVA_main(Gama_Update[0])
    BVA_acc = UPDATE.accuracy
    BVA_isize = UPDATE.total_inject_size

    return [BVA_acc, BVA_isize]

def plot_figure():
    with open(os.path.join(utils.RESULT_PATH, 'ActiveUpdateModifiedRer{}.pkl'.format(update_dis)), 'rb') as f:
        Gamma_of_BVA, BVA_final_acc, BVA_final_isize = pickle.load(f)
        f.close()
    print(Gamma_of_BVA)
    print(BVA_final_acc[len(BVA_final_acc)-3])

    plt.rcParams.update({
        "legend.fancybox": False,
        "legend.frameon": True,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"], #注意这里是Times，不是Times New Roman
        "font.size":20})
    plt.figure()
    ax=plt.subplot()
    ax.grid()    
    ax.set_ylim(-0.05,1.05)

    acc1 = BVA_final_acc[0]
    acc2 = BVA_final_acc[2]
    acc3 = BVA_final_acc[4]
    acc4 = BVA_final_acc[6]

    BVA_avr_acc1 = np.mean(acc1, axis=1)
    BVA_min_acc1 = BVA_avr_acc1 - np.min(acc1, axis=1)
    BVA_max_acc1 = np.max(acc1, axis=1) - BVA_avr_acc1

    BVA_avr_acc2 = np.mean(acc2, axis=1)
    BVA_min_acc2 = BVA_avr_acc2 - np.min(acc2, axis=1)
    BVA_max_acc2 = np.max(acc2, axis=1) - BVA_avr_acc2

    BVA_avr_acc3 = np.mean(acc3, axis=1)
    BVA_min_acc3 = BVA_avr_acc3 - np.min(acc3, axis=1)
    BVA_max_acc3 = np.max(acc3, axis=1) - BVA_avr_acc3

    BVA_avr_acc4 = np.mean(acc4, axis=1)
    BVA_min_acc4 = BVA_avr_acc4 - np.min(acc4, axis=1)
    BVA_max_acc4 = np.max(acc4, axis=1) - BVA_avr_acc4

    ax.errorbar(UP_PER, BVA_avr_acc1, yerr=[BVA_min_acc1, BVA_max_acc1], 
        marker = '>', 
        ecolor='purple', color = 'plum', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'purple', markersize = 10, markeredgewidth=0.8, label = r'$\gamma = 0.5\#W$'
        )

    ax.errorbar(UP_PER, BVA_avr_acc2, yerr=[BVA_min_acc2, BVA_max_acc2], 
        marker = 'o', 
        ecolor='blue', color = 'lightblue', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'blue', markersize = 10, markeredgewidth=0.8, label = r'$\gamma = 2\#W$'
        )

    ax.errorbar(UP_PER, BVA_avr_acc3, yerr=[BVA_min_acc3, BVA_max_acc3], 
        marker = 's', 
        ecolor='green', color = 'lightgreen', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'lightgreen', markersize = 10, markeredgewidth=0.8, label = r'$\gamma = 8\#W$'
        )

    ax.errorbar(UP_PER, BVA_avr_acc4, yerr=[BVA_min_acc4, BVA_max_acc4], 
        marker = '<', 
        ecolor='red', color = 'salmon', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = r'$\gamma = 32\#W$'
        )
    #ax.legend(loc = (0.54, 0.54))
    ax.legend()
    ax.set_xlabel(r'UP ($\%$)')
    #ax.set_xlabel(r'Update Percentage ($\%$, #Upd/#TotF)')
    ax.set_ylabel('Recovery rate')
    plt.savefig(utils.PLOTS_PATH + '/' + 'UpdateAccWithUpdate{}.pdf'.format(update_dis), bbox_inches = 'tight', dpi = 600)


    plt.show()    

if __name__=='__main__':
    
    d_id2 = input("input update operations: 1. All Add 2. Uniform 3. All Delete ")
    if d_id2=='1':
        update_dis = 'AllAdd'
    elif d_id2=='2':
        update_dis = 'Uniform'
    else:
        update_dis = 'AllDelete'
    kw_to_id = utils.get_kws_id(chosen_kws)
    #print(len(chosen_kws))
    # gamma_of_BVA = 500000
    exp_times = 10
    Gamma_of_BVA = []
    min_gamma = (int)(len(chosen_kws)/2)
    Gamma_of_BVA.append(min_gamma)
    while(Gamma_of_BVA[len(Gamma_of_BVA)-1]<96000):
        Gamma_of_BVA.append(Gamma_of_BVA[len(Gamma_of_BVA)-1]*2)
    UP_PER = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #, 10, 50, 100, 1000
    
    BVA_final_acc = [[[0]]*len(UP_PER) for _ in range(len(Gamma_of_BVA))]#[0]*len(Gamma_of_BVA)
    BVA_final_isize = [0]*len(Gamma_of_BVA)
    print(BVA_final_acc)

    for ind in range(len(Gamma_of_BVA)):
        for ind2 in range(len(UP_PER)):
            tmptmp = []
            Gama_Update_List = [(Gamma_of_BVA[ind], UP_PER[ind2])]*30
            partial_function = partial(process, kw_dict, chosen_kws, docu, adv_observed_offset, observed_period, target_period, number_queries_per_period, kw_to_id, update_dis)
            with Pool(processes=exp_times) as pool:
                for result in pool.map(partial_function, Gama_Update_List):
                    #BVA_final_acc[ind][ind2].append(result[0])
                    tmptmp.append(result[0])
                    BVA_final_isize[ind] = result[1]
       
            BVA_final_acc[ind][ind2] = tmptmp
    print(Gamma_of_BVA)
    print(BVA_final_acc)
    print(BVA_final_isize)
    with open(os.path.join(utils.RESULT_PATH, 'ActiveUpdateModifiedRer{}.pkl'.format(update_dis)), 'wb') as f:
        pickle.dump((Gamma_of_BVA, BVA_final_acc, BVA_final_isize), f)
        f.close()
    plot_figure()
    


