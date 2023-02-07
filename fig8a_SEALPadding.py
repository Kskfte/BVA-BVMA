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
    real_size, real_length, offset_of_Decoding = pickle.load(f)
    f.close()
exp_times = 10
offset_of_Decoding_list = [offset_of_Decoding]*exp_times

class Attack: 
    def __init__(self, doc, min_file_size, max_file_size, SEALx, chosen_kws, observed_queries, target_queries, kws_leak_percent, trend_matrix_norm, real_size, real_length, offset_of_Decoding):
        self.real_tag = {}
        self.recover_tag = {}

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

        self.bandwidth_setup = 0 #setup database
        self.bandwidth_update = 0 #a file contains multiple keywords, BVA, BVMA ;no padding of Decoding, impractical of SR.
        self.bandwidth_query = 0 #return No. of files

        self.kws_leak_percent = kws_leak_percent
        self.chosen_kws = chosen_kws
        self.target_queries = target_queries
        self.observed_queries = observed_queries
        self.trend_matrix_norm = trend_matrix_norm
        
        self.SEALx = SEALx
  
        self.size_without_padding, self.length_without_padding = real_size, real_length 
        self.size_after_setup_padding, self.length_after_setup_padding = {}, {}
        #self.size_after_injection_padding, self.length_after_injection_padding = {}, {}
        self.min_file_size, self.max_file_size = min_file_size, max_file_size
        self.injection_length_without_padding = {} 
        self.length_after_injection_without_padding = {}
        self.size_after_injection_without_padding = {}
        """
        baseline phase
        """
        #self.observed_size, self.max_observed_size, self.observed_length = self.get_baseline_observed_size_and_length(real_size, real_length)
        """
        get offset of Decoding
        """
        self.offset = offset_of_Decoding
        #self.Group = self.Group_cluster()
    
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
        
        self.size_dynamic_query = (int) (sum(self.length_after_injection_without_padding.values())/len(chosen_kws))
        self.bandwidth_dynamic_query = sum(self.length_after_injection_without_padding.values()) / (sum(self.length_without_padding.values()) + sum(self.injection_length_without_padding.values())) - 1
        
        number_of_padding_injection = 0
        for k in self.length_after_injection_without_padding.keys():
            tmp = self.length_after_injection_without_padding[k]-self.injection_length_without_padding[k]-self.length_after_setup_padding[k]
            if tmp>number_of_padding_injection:
                number_of_padding_injection = tmp
        
        self.bandwidth_injection = number_of_padding_injection/math.ceil(math.log2(len(chosen_kws)))
        self.size_injection = (number_of_padding_injection + math.ceil(math.log2(len(chosen_kws))))

    def get_size_and_length_after_setup_padding(self):
        """
        Padding of setup phase
        """
        self.size_after_setup_padding = {}
        self.length_after_setup_padding = {}
        self.size_after_setup_padding, self.length_after_setup_padding = utils.get_kws_size_and_length_after_padding(self.doc, self.chosen_kws, self.SEALx)

    def get_baseline_observed_size_and_length(self):
        """
        observe size and length in baseline phase
        """
        observed_size = {}
        observed_length = {}
        max_observed_size = 0
        for i_week in self.observed_queries:
            for query in i_week:
                observed_size[query] = self.size_after_setup_padding[query]
                observed_length[query] = self.length_after_setup_padding[query]
                if max_observed_size < observed_size[query]:
                    max_observed_size = observed_size[query]
        return observed_size, max_observed_size, observed_length
    def get_size_and_length_after_injection_without_padding(self, injection_length, injection_size):
        self.length_after_injection_without_padding = {}
        self.size_after_injection_without_padding = {}
        for k in self.length_after_setup_padding.keys():
            if k in injection_length.keys():
                self.length_after_injection_without_padding[k] = self.length_after_setup_padding[k] + injection_length[k]
                self.size_after_injection_without_padding[k] = self.size_after_setup_padding[k] + injection_size[k]

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
                    if query in self.size_after_injection_without_padding.keys(): 
                        if (self.size_after_injection_without_padding[query] - observed_size_in_setup[kw_id]) % self.gamma == 0:
                            self.recover_tag[query] = (self.size_after_injection_without_padding[query] - observed_size_in_setup[kw_id]) / self.gamma
                            break
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
        self.get_size_and_length_after_injection_without_padding(injection_length, injection_size)

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
            
        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                CA = []
                findflag = False
                for kw_id in observed_size_in_setup.keys():
                    if query in self.size_after_injection_without_padding.keys() and self.size_after_injection_without_padding[query] - observed_size_in_setup[kw_id] >= 0 and (self.size_after_injection_without_padding[query] - observed_size_in_setup[kw_id]) - (len(self.chosen_kws)/2)*math.log2(len(self.chosen_kws)) < len(self.chosen_kws):                           
                        diffReBa = (int) ((self.size_after_injection_without_padding[query] - observed_size_in_setup[kw_id])/1)
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
                                if self.length_after_injection_without_padding[query] - observed_length_in_setup[kw_id] == num_tF:
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
        self.get_size_and_length_after_injection_without_padding(injection_length, injection_size)
def multiprocess_worker(kw_dict, chosen_kws, docu, xx, adv_observed_offset, observed_period, target_period, number_queries_per_period, real_size, real_length, offset_of_Decoding):

    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), 260, adv_observed_offset)
    begin_time = random.randint(0, len(trend_matrix[0])-observed_period-target_period-1)
    observed_queries = utils.generate_queries(trend_matrix[:, begin_time:begin_time+observed_period], 'real-world', number_queries_per_period)
    target_queries = utils.generate_queries(trend_matrix[:, begin_time+observed_period:begin_time+observed_period+target_period], 'real-world', number_queries_per_period)

    min_file_size, max_file_size, block_size = utils.get_file_size(docu)
    seal = Attack(docu, min_file_size, max_file_size, xx, chosen_kws, observed_queries, target_queries, 1, trend_matrix, real_size, real_length, offset_of_Decoding)

    seal.get_bandwidth()
    BVA_g = (int) (block_size*math.ceil(len(chosen_kws)/(2*block_size)))
    seal.BVA_main(BVA_g) 
    BVA_acc = seal.accuracy
    #print(seal.accuracy)
    seal.BVMA_SP_main('real-world') 
    BVMA_acc = seal.accuracy
    #print(seal.accuracy)
    return [BVA_acc, BVMA_acc, seal.bandwidth_setup, seal.size_setup, seal.bandwidth_static_query, seal.size_static_query, 
        seal.bandwidth_injection, seal.size_injection, seal.bandwidth_dynamic_query, seal.size_dynamic_query]

def plot_fiure(BVA_acc, BVMA_acc, Bandwidth_setup, Bandwidth_static_query, Bandwidth_update, Bandwidth_dynamic_query):

    labels = ['x=0', 'x=2', 'x=4', 'x=16']
    c = []
    for i in range(len(BVA_acc)):
        for j in range(len(BVA_acc[0])):
            c.append(['BVA', labels[i], BVA_acc[i][j]])
            c.append(['BVMA', labels[i], BVMA_acc[i][j]])


    df = pd.DataFrame(c, columns=['Attack-Rer', '', 'Recovery rate']) 
    print(df)
    bandwidth_setup = np.mean(Bandwidth_setup, axis=1)
    bandwidth_static_query = np.mean(Bandwidth_static_query, axis=1)
    bandwidth_update = np.mean(Bandwidth_update, axis=1)
    bandwidth_dynamic_query = np.mean(Bandwidth_dynamic_query, axis=1)

    band = []
    for i in range(len(bandwidth_setup)):
        band.append(['Setup', labels[i], bandwidth_setup[i]*100])
        band.append(['S-Query', labels[i], bandwidth_static_query[i]*100])
        band.append(['Injection', labels[i], bandwidth_update[i]*100])
        band.append(['I-Query', labels[i], bandwidth_dynamic_query[i]*100])

    plt.clf()
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], 
    "font.size":30,
    "lines.markersize":20})

    df2 = pd.DataFrame(band, columns=['Phase-Overhead', '', r'Overhead ($\%$)']) 

    _, ax = plt.subplots()
    pale = {"BVA": 'skyblue', "BVMA": 'red'}
    sns.boxplot(x = '', y = 'Recovery rate', hue = 'Attack-Rer',data=df, palette=pale, width=0.25,linewidth=1)
    ax.set_ylim(0.65, 0.9)
    ax.legend(loc = 2, bbox_to_anchor = (-0.9, 1.1))
    ax2 = ax.twinx()
    markers = {'Setup': 'X', 'S-Query': '<', 'Injection': 'o', 'I-Query': '>'}
    sns.scatterplot(x='', y=r'Overhead ($\%$)', style = 'Phase-Overhead', hue = 'Phase-Overhead', s = 200, markers=markers,  ax=ax2, data=df2)
    XX = [0, 1, 2, 3]
    ax2.plot(XX, [i*100 for i in bandwidth_setup],  '--b')
    ax2.plot(XX, [i*100 for i in bandwidth_static_query], color='orange',ls='--')
    ax2.plot(XX, [i*100 for i in bandwidth_update], '--g')
    ax2.plot(XX, [i*100 for i in bandwidth_dynamic_query], color='salmon',ls='--')
    ax2.legend(loc = 2, bbox_to_anchor = (-0.9, 0.7))

    plt.savefig(utils.PLOTS_PATH + '/' + 'SEALEnronPaddingORAM.pdf', bbox_inches = 'tight', dpi = 600)
    plt.show()

if __name__=='__main__': 
   
    SEALx = [0, 2, 4, 16]
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
    pbar = tqdm(total=len(SEALx))
    loop = 0
    for xx in SEALx:
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
        partial_function = partial(multiprocess_worker, kw_dict, chosen_kws, docu, xx, adv_observed_offset, observed_period, target_period, number_queries_per_period, real_size, real_length)
        with Pool(processes=exp_times) as pool:
            for result in pool.map(partial_function, offset_of_Decoding_list):
                BVA_tmp_acc.append(result[0])
                BVMA_tmp_acc.append(result[1])
                Bandwidth_tmp_setup.append(result[2])
                Size_tmp_setup.append(result[3])
                Bandwidth_tmp_static_query.append(result[4])
                Size_tmp_static_query.append(result[5])
                Bandwidth_tmp_injection.append(result[6])
                Size_tmp_injection.append(result[7])
                Bandwidth_tmp_dynamic_query.append(result[8])
                Size_tmp_dynamic_query.append(result[9])

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

        pbar.update(math.ceil((loop+1)/len(SEALx)))
        loop += 1
    pbar.close()
    with open(os.path.join(utils.RESULT_PATH, 'SEALEnronPaddingORAM.pkl'), 'wb') as f:
        pickle.dump((BVA_acc, BVMA_acc, Bandwidth_setup, Size_setup, Bandwidth_static_query, Size_static_query, Bandwidth_injection, Size_injection, Bandwidth_dynamic_query, Size_dynamic_query), f)
        f.close()

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
    plot_fiure(BVA_acc, BVMA_acc, Bandwidth_setup, Bandwidth_static_query, Bandwidth_injection, Bandwidth_dynamic_query)
