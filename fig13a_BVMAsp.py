import math
import os
import pickle
import utils
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import attacks
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import random

def multiprocess_worker(chosen_kws, real_size, real_length, offset_of_Decoding, trend_matrix, observed_period, target_period, number_queries_per_period, Kws_Gama):

    begin_time = random.randint(0, len(trend_matrix[0])-observed_period-target_period-1)
    random.shuffle(chosen_kws)
    observed_queries = utils.generate_queries(trend_matrix[:, begin_time:begin_time+observed_period], 'real-world', number_queries_per_period)
    target_queries = utils.generate_queries(trend_matrix[:, begin_time+observed_period:begin_time+observed_period+target_period], 'real-world', number_queries_per_period)
    attack = attacks.Attack(chosen_kws, observed_queries, target_queries, Kws_Gama, trend_matrix, real_size, real_length, 0)

 
    attack.BVMA_NoSP_main()
    BVMA_accuracy = (attack.accuracy)
    BVMA_injection_length = (attack.total_inject_length)
    BVMA_injection_size = (attack.total_inject_size)
    attack.BVMA_SP_main('real-world')
    BVMA_SP_accuracy = (attack.accuracy)
    BVMA_SP_injection_length = (attack.total_inject_length)
    BVMA_SP_injection_size = (attack.total_inject_size)
    
    return [BVMA_accuracy, BVMA_injection_length, BVMA_injection_size,
            BVMA_SP_accuracy, BVMA_SP_injection_length, BVMA_SP_injection_size]

def plot_figure(dataset_name):
    with open(utils.RESULT_PATH + '/' + 'BVMASP{}.pkl'.format(dataset_name), 'rb') as f:
        (BVMA_accuracy, BVMA_SP_accuracy) = pickle.load(f)
        f.close()
  
    plt.clf()
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], 
    "font.size":27,
    "lines.markersize":18})
    # -*- coding: utf-8 -*
    
    X=[r'BVMA$\_$SP',r'BVMA$\_$NoSP']
    Y = [BVMA_SP_accuracy[0], BVMA_accuracy[0]]
    plt.figure()
    ax=plt.subplot()
    ax.bar(X,Y,width=0.17, edgecolor = ["red", "blue"], color=["salmon", "skyblue"])
    
    y_major_locator = MultipleLocator(0.2)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Recovery rate")
    plt.grid(axis='y')
    plt.savefig(utils.PLOTS_PATH + '/' + 'BVMASP{}.pdf'.format(dataset_name), bbox_inches = 'tight', dpi = 600)
    plt.show()

if __name__=='__main__': 
   
    if not os.path.exists(utils.RESULT_PATH):
        os.makedirs(utils.RESULT_PATH)
    if not os.path.exists(utils.PLOTS_PATH):
        os.makedirs(utils.PLOTS_PATH)
    """ choose dataset """
    dataset_name = 'Enron'
    number_queries_per_period = 1000
    observed_period = 8
    target_period = 10
    adv_observed_offset = 10
    
    # plot_figure(dataset_name)
    """ read data """
    with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
    chosen_kws = list(kw_dict.keys())
    with open(os.path.join(utils.DATASET_PATH, "{}_wl_v_off.pkl".format(dataset_name.lower())), "rb") as f:
        real_size, real_length, offset_of_Decoding = pickle.load(f)
        f.close()
    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), 260, adv_observed_offset)

    """ experiment parameter """
    exp_times = 10 
    
    kws_leak_percent = [1]


    BVMA_accuracy = [0]*len(kws_leak_percent)
    BVMA_injection_length = [0]*len(kws_leak_percent)
    BVMA_injection_size = [0]*len(kws_leak_percent)
    BVMA_SP_accuracy = [0]*len(kws_leak_percent)
    BVMA_SP_injection_length = [0]*len(kws_leak_percent)
    BVMA_SP_injection_size = [0]*len(kws_leak_percent)
    pbar = tqdm(total=len(kws_leak_percent))
    for ind in range(len(kws_leak_percent)):
        partial_function = partial(multiprocess_worker, chosen_kws, real_size, real_length, offset_of_Decoding, trend_matrix, observed_period, target_period, number_queries_per_period)
        with Pool(processes=exp_times) as pool:
            for result in pool.map(partial_function, [kws_leak_percent[ind]]*exp_times):
                BVMA_accuracy[ind] += result[0]
                BVMA_injection_length[ind] += result[1]
                BVMA_injection_size[ind] += result[2]
                BVMA_SP_accuracy[ind] += result[3]
                BVMA_SP_injection_length[ind] += result[4]
                BVMA_SP_injection_size[ind] += result[5]
            BVMA_accuracy[ind] /= exp_times
            BVMA_injection_length[ind] /= exp_times
            BVMA_injection_size[ind] /= exp_times
            BVMA_SP_accuracy[ind] /= exp_times
            BVMA_SP_injection_length[ind] /= exp_times
            BVMA_SP_injection_size[ind] /= exp_times
        pbar.update(math.ceil((ind+1)/len(kws_leak_percent)))
    pbar.close()
    """ save result """
    print(BVMA_accuracy)
    print(BVMA_SP_accuracy)
    SaveResult = (BVMA_accuracy, BVMA_SP_accuracy)
    with open(utils.RESULT_PATH + '/' + 'BVMASP{}.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump(SaveResult, f)
        f.close()

    """ plot figure """
    plot_figure(dataset_name)

    
