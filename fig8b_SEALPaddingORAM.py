from audioop import avg
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
exp_times = 30
offset_of_Decoding_list = [offset_of_Decoding]*exp_times

class Attack: 
    def __init__(self, doc, min_file_size, max_file_size, block_size, SEALx, chosen_kws, observed_queries, target_queries, kws_leak_percent, trend_matrix_norm, real_size, real_length, offset_of_Decoding):
        self.real_tag = {}
        self.recover_tag = {}

        self.doc = doc

        self.time = 0
        self.inject_time = 0        
        self.recover_queries_num = 0
        self.total_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0

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
        self.block_size = block_size
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
    
    def get_size_and_length_after_setup_padding(self):
        """
        Padding of setup phase
        """
        self.size_after_setup_padding = {}
        self.length_after_setup_padding = {}
        self.size_after_setup_padding, self.length_after_setup_padding = utils.get_kws_size_and_length_after_seal(self.doc, self.chosen_kws, self.SEALx, self.block_size)

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
                # self.length_after_injection_without_padding[k] = self.length_after_setup_padding[k] + injection_length[k]
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
        #print("BVARecoeryTime:{}".format(self.time))
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
        self.get_size_and_length_after_injection_without_padding(injection_length, injection_size)

def multiprocess_worker(kw_dict, chosen_kws, docu, xx, adv_observed_offset, observed_period, target_period, number_queries_per_period, real_size, real_length, offset_of_Decoding):

    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), 260, adv_observed_offset)
    begin_time = random.randint(0, len(trend_matrix[0])-observed_period-target_period-1)
    observed_queries = utils.generate_queries(trend_matrix[:, begin_time:begin_time+observed_period], 'real-world', number_queries_per_period)
    target_queries = utils.generate_queries(trend_matrix[:, begin_time+observed_period:begin_time+observed_period+target_period], 'real-world', number_queries_per_period)


    min_file_size, max_file_size, block_size = utils.get_file_size(docu)
    seal = Attack(docu, min_file_size, max_file_size, block_size, xx, chosen_kws, observed_queries, target_queries, 1, trend_matrix, real_size, real_length, offset_of_Decoding)

    BVA_g = (int) (block_size*math.ceil(len(chosen_kws)/(2*block_size)))
    seal.BVA_main(BVA_g) 
    BVA_acc = seal.accuracy
    return BVA_acc

def plot_fiure(BVA_acc):
    
    labels = ['x=0', 'x=2', 'x=4', 'x=16']
    c = []
    for i in range(len(BVA_acc)):
        for j in range(len(BVA_acc[0])):
            c.append(['BVA', labels[i], BVA_acc[i][j]])


    df = pd.DataFrame(c, columns=['Rer', '', 'Recovery rate']) 

    plt.clf()
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], 
    "font.size":30,
    "lines.markersize":20})

    pale = {"BVA": 'skyblue'}
    sns.boxplot(x = '', y = 'Recovery rate', hue = 'Rer',data=df, palette=pale, width=0.25,linewidth=1)

    plt.savefig(utils.PLOTS_PATH + '/' + 'SEALPaddingEnron.pdf', bbox_inches = 'tight', dpi = 600)
    plt.show()

if __name__=='__main__': 
   
    SEALx = [0, 2, 4, 16]
    BVA_acc = []
    pbar = tqdm(total=len(SEALx))
    loop = 0
    for xx in SEALx:
        BVA_tmp_acc = []
        partial_function = partial(multiprocess_worker, kw_dict, chosen_kws, docu, xx, adv_observed_offset, observed_period, target_period, number_queries_per_period, real_size, real_length)
        with Pool(processes=exp_times) as pool:
            for result in pool.map(partial_function, offset_of_Decoding_list):
                BVA_tmp_acc.append(result)
        BVA_acc.append(BVA_tmp_acc)
        pbar.update(math.ceil((loop+1)/len(SEALx)))
        loop += 1
    pbar.close()
    with open(os.path.join(utils.RESULT_PATH, 'SEALPaddingEnron.pkl'), 'wb') as f:
        pickle.dump(BVA_acc, f)
        f.close()

    print(BVA_acc)
    plot_fiure(BVA_acc)