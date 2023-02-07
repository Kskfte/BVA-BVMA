import pickle
import matplotlib.pyplot as plt
import os
import utils
import numpy as np
from matplotlib.pyplot import MultipleLocator
from tqdm import tqdm
from time import sleep
import math

def get_new_queries(kw_dict, number_queries_per_periods, maximum_observed_periods, adv_observed_offset):
    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), maximum_observed_periods, adv_observed_offset)
    queries = utils.generate_queries(trend_matrix, 'real-world', number_queries_per_periods)
    ObservedTimes = []
    for i in range(maximum_observed_periods):
        ObservedTimes.append(i+1)
    RealWorldNewQueries = []
    TrendTotalQueries = {}
    for Q in queries:
        new_query_count = 0
        for q in Q:
            if q not in TrendTotalQueries.keys():
                new_query_count += 1
                TrendTotalQueries[q] = 0
        RealWorldNewQueries.append(new_query_count)

    queries = utils.generate_queries(trend_matrix, 'uniform', number_queries_per_periods)
    UniformNewQueries = []
    UniformTotalQueries = {}
    for Q in queries:
        new_query_count = 0
        for q in Q:
            if q not in UniformTotalQueries.keys():
                new_query_count += 1
                UniformTotalQueries[q] = 0
        UniformNewQueries.append(new_query_count)

    return ObservedTimes, RealWorldNewQueries, UniformNewQueries

def plot_figure(ObservedTimes, RealWorldNewQueries, UniformNewQueries, dataset_name):
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], #注意这里是Times，不是Times New Roman
    "font.size":20})
    _, ax = plt.subplots()
    plt.plot(ObservedTimes, RealWorldNewQueries, 'r-', label = 'real-world')
    plt.plot(ObservedTimes, UniformNewQueries, 'g--', label = 'uniform')
    plt.grid() 
    plt.legend(loc = 'upper right')
    plt.tick_params()
    if dataset_name=='Wiki':
        x_major_locator=MultipleLocator(16) 
        plt.xlabel('Observed months')
    else:   
        x_major_locator=MultipleLocator(4) 
        plt.xlabel('Observed weeks')
    plt.ylabel('New queries')
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig(utils.PLOTS_PATH + '/' + 'NewQueries{}.pdf'.format(dataset_name), bbox_inches = 'tight', dpi = 600)
    plt.show()

if __name__ == '__main__':

    if not os.path.exists(utils.RESULT_PATH):
        os.makedirs(utils.RESULT_PATH)
    if not os.path.exists(utils.PLOTS_PATH):
        os.makedirs(utils.PLOTS_PATH)
    """ choose dataset """
    d_id = input("input evaluation dataset: 1. Enron 2. Lucene 3. WikiPedia ")
    dataset_name = ''
    print(d_id)
    exp_times = 10
    number_queries_per_periods = 1000
    maximum_observed_periods = 20
    adv_observed_offset = 10
    if d_id=='1':
        dataset_name = 'Enron'
    elif d_id=='2':
        dataset_name = 'Lucene'  
    elif d_id=='3':
        dataset_name = 'Wiki'
        number_queries_per_periods = 5000
        maximum_observed_periods = 50
    else:
        raise ValueError('No Selected Dataset!!!')
    
    ObservedTimes = [0]*maximum_observed_periods
    RealWorldNewQueries = [0]*maximum_observed_periods
    UniformNewQueries = [0]*maximum_observed_periods
    with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
    pbar = tqdm(total=100)
    for _ in range(exp_times):
        Ob, Tr, Uni = get_new_queries(kw_dict, number_queries_per_periods, maximum_observed_periods, adv_observed_offset)
        ObservedTimes = np.sum([Ob,ObservedTimes], axis=0)
        RealWorldNewQueries = np.sum([Tr,RealWorldNewQueries], axis=0)
        UniformNewQueries = np.sum([Uni,UniformNewQueries], axis=0)
        sleep(0.1)
        pbar.update(math.ceil(100/exp_times))
    pbar.close()
    print("Observed periods: {}".format(np.divide(ObservedTimes,exp_times)))
    print("RealWorldNewQueries: {}".format(np.divide(RealWorldNewQueries,exp_times)))
    print("UniformNewQueries: {}".format(np.divide(UniformNewQueries,exp_times)))
    plot_figure(np.divide(ObservedTimes,exp_times), np.divide(RealWorldNewQueries,exp_times), np.divide(UniformNewQueries,exp_times), dataset_name)



