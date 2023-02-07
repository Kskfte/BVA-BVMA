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
    attack = attacks.Attack(chosen_kws, observed_queries, target_queries, Kws_Gama[0], trend_matrix, real_size, real_length, offset_of_Decoding)

    if Kws_Gama[1] == (int) (offset_of_Decoding/4):
        attack.Decoding_main()
    Decoding_accuracy = (attack.accuracy)
    Decoding_injection_length = (attack.total_inject_length)
    Decoding_injection_size = (attack.total_inject_size)
    if Kws_Gama[1] == (int) (offset_of_Decoding/4):
        attack.BVMA_SP_main('real-world')
    BVMA_accuracy = (attack.accuracy)
    BVMA_injection_length = (attack.total_inject_length)
    BVMA_injection_size = (attack.total_inject_size)
    if Kws_Gama[1] == (int) (offset_of_Decoding/4):        
        attack.SR_main(1)
    SR_1_accuracy = (attack.accuracy)
    SR_1_injection_length = (attack.total_inject_length)
    SR_1_injection_size = (attack.total_inject_size)
    if Kws_Gama[1] == (int) (offset_of_Decoding/4):
        attack.SR_main(len(chosen_kws))
    SR_linear_accuracy = (attack.accuracy)
    SR_linear_injection_length = (attack.total_inject_length)
    SR_linear_injection_size = (attack.total_inject_size)
    attack.BVA_main(Kws_Gama[1])
    BVA_accuracy = (attack.accuracy)
    BVA_injection_length = (attack.total_inject_length)
    BVA_injection_size = (attack.total_inject_size)
    return [Decoding_accuracy, Decoding_injection_length, Decoding_injection_size, 
        BVA_accuracy, BVA_injection_length, BVA_injection_size,
        BVMA_accuracy, BVMA_injection_length, BVMA_injection_size,
        SR_1_accuracy, SR_1_injection_length, SR_1_injection_size,     
        SR_linear_accuracy, SR_linear_injection_length, SR_linear_injection_size,
        attack.Decoding_rer_time, attack.SR_min_rer_time, attack.SR_max_rer_time,
        attack.BVMA_rer_time, attack.BVA_min_rer_time, attack.BVA_max_rer_time]


def plot_figure(dataset_name):
    with open(utils.RESULT_PATH + '/' + 'AttacksWithKwsLeak{}.pkl'.format(dataset_name), 'rb') as f:
        (kws_leak_percent,  
        Decoding_accuracy, Decoding_injection_length, Decoding_injection_size, 
        BVA_accuracy, BVA_injection_length, BVA_injection_size,
        BVMA_accuracy, BVMA_injection_length, BVMA_injection_size,
        SR_1_accuracy, SR_1_injection_length, SR_1_injection_size, 
        SR_linear_accuracy, SR_linear_injection_length, SR_linear_injection_size) = pickle.load(f)
        f.close()
    BVA_avr_acc = np.mean(BVA_accuracy, axis=0)
    BVA_min_acc = BVA_avr_acc - np.min(BVA_accuracy, axis=0)
    BVA_max_acc = np.max(BVA_accuracy, axis=0) - BVA_avr_acc
    BVA_injection_length = BVA_injection_length[len(BVA_injection_length)-1]
    BVA_avr_size = np.mean(BVA_injection_size, axis=0)
    BVA_min_size = BVA_avr_size - np.min(BVA_injection_size, axis=0)
    BVA_max_size = np.max(BVA_injection_size, axis=0) - BVA_avr_size
    print(BVA_avr_acc)
    print(BVA_min_acc)
    print(BVA_max_acc)
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], #注意这里是Times，不是Times New Roman
    "font.size":20})
    plt.figure()
    ax=plt.subplot()
      
    y_major_locator=MultipleLocator(0.2)
    ax.yaxis.set_major_locator(y_major_locator)
    #ax.set_ylim(-0.05,1.0)
    ax.errorbar(kws_leak_percent, SR_1_accuracy, 
        marker = '>', 
        ecolor='purple', color = 'plum', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'purple', markersize = 10, markeredgewidth=0.8, label = 'Single-round:m=1'
        )
    ax.errorbar(kws_leak_percent, SR_linear_accuracy, zorder=1, 
        marker = '<',
        ecolor='purple', color = 'plum', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'purple', markersize = 10, markeredgewidth=0.8, label = 'Single-round:m=$\#$W'
        )
    ax.errorbar(kws_leak_percent, Decoding_accuracy, zorder=2, color = 'lightsalmon', marker = 'x', markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = 'Decoding')
    
    ax.errorbar(kws_leak_percent, BVA_avr_acc, zorder=3,
            yerr=[BVA_min_acc, BVA_max_acc],  marker = 'o', 
            ecolor='green', color = 'lightgreen', elinewidth=1, capsize=4, markeredgecolor = 'green', markersize = 10, markeredgewidth=0.8, label = 'BVA'
        )
    
    ax.errorbar(kws_leak_percent, BVMA_accuracy, zorder=4, color = 'lightblue', marker = '2', markeredgecolor = 'blue', markersize = 10, markeredgewidth=0.8, label = 'BVMA')
   
    plt.xlabel('Kws leakage rate')
    plt.ylabel('Recovery rate')
    plt.grid()
    plt.legend()
    plt.savefig(utils.PLOTS_PATH + '/' + 'Rer{}.pdf'.format(dataset_name), bbox_inches = 'tight', dpi = 600)


    plt.figure()
    
    ax=plt.subplot()

    ax.errorbar(kws_leak_percent, SR_1_injection_size, 
        marker = '>',
        ecolor='purple', color = 'plum', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'purple', markersize = 10, markeredgewidth=0.8, label = 'Single-round:m=1'
        )
    ax.errorbar(kws_leak_percent, SR_linear_injection_size, 
        marker = '<',
        ecolor='purple', color = 'plum', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'purple', markersize = 10, markeredgewidth=0.8, label = 'Single-round:m=$\#$W'
        )
    ax.errorbar(kws_leak_percent, Decoding_injection_size, color = 'lightsalmon', marker = 'x', markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = 'Decoding')
    
    ax.errorbar(kws_leak_percent, BVA_avr_size, 
            yerr=[BVA_min_size, BVA_max_size],  marker = 'o', 
            ecolor='green', color = 'lightgreen', elinewidth=1, capsize=4, markeredgecolor = 'green', markersize = 10, markeredgewidth=0.8, label = 'BVA'
        )
    ax.errorbar(kws_leak_percent, BVMA_injection_size, color = 'lightblue', marker = '2', markeredgecolor = 'blue', markersize = 10, markeredgewidth=0.8, label = 'BVMA')
    
    plt.yscale('log')
    plt.xlabel('Kws leakage rate')
    plt.ylabel('Injection size')
    plt.grid()
    plt.legend()
    plt.savefig(utils.PLOTS_PATH + '/' + 'Size{}.pdf'.format(dataset_name), bbox_inches = 'tight', dpi = 600)

    plt.figure()
    
    ax=plt.subplot()
    ax.errorbar(kws_leak_percent, SR_1_injection_length, 
        marker = '>',
        ecolor='purple', color = 'plum', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'purple', markersize = 10, markeredgewidth=0.8, label = 'Single-round:m=1'
        )
    ax.errorbar(kws_leak_percent, SR_linear_injection_length, 
        marker = '<',
        ecolor='purple', color = 'plum', mfc = 'white', elinewidth=1, capsize=4, markeredgecolor = 'purple', markersize = 10, markeredgewidth=0.8, label = 'Single-round:m=$\#$W'
        )
    ax.errorbar(kws_leak_percent, Decoding_injection_length, color = 'lightsalmon', marker = 'x', markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = 'Decoding')
    ax.errorbar(kws_leak_percent, BVA_injection_length, color = 'lightgreen', marker = 'o', markeredgecolor = 'green', markersize = 10, markeredgewidth=0.8, label = 'BVA')
    ax.errorbar(kws_leak_percent, BVMA_injection_length, color = 'lightblue', marker = '2', markeredgecolor = 'blue', markersize = 10, markeredgewidth=0.8, label = 'BVMA')
    
    plt.yscale('log')
    plt.xlabel('Kws leakage rate')
    plt.ylabel('Injection length')
    plt.grid()
    plt.legend(loc='center', bbox_to_anchor=(0.5, 0.65))
    plt.savefig(utils.PLOTS_PATH + '/' + 'Length{}.pdf'.format(dataset_name), bbox_inches = 'tight', dpi = 600)
    plt.show()

if __name__=='__main__': 
   
    if not os.path.exists(utils.RESULT_PATH):
        os.makedirs(utils.RESULT_PATH)
    if not os.path.exists(utils.PLOTS_PATH):
        os.makedirs(utils.PLOTS_PATH)
    """ choose dataset """
    d_id = input("input evaluation dataset: 1. Enron 2. Lucene 3.WikiPedia ")
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
    with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
    chosen_kws = list(kw_dict.keys())
    with open(os.path.join(utils.DATASET_PATH, "{}_wl_v_off.pkl".format(dataset_name.lower())), "rb") as f:
        real_size, real_length, offset_of_Decoding = pickle.load(f)#_after_padding_2
        f.close()
    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), 260, adv_observed_offset)

    """ experiment parameter """
    exp_times = 1 #change experimental times
    BVA_gamma_list = []
    minimum_gamma = (int) (len(kw_dict)/2)
    maximum_gamma = (int) (offset_of_Decoding/4)
    BVA_gamma_list.append(minimum_gamma)
    while minimum_gamma<maximum_gamma/2:
        minimum_gamma *= 2
        minimum_gamma += 1
        BVA_gamma_list.append(minimum_gamma)
    BVA_gamma_list.append(maximum_gamma)
    print(len(BVA_gamma_list))
    kws_leak_percent = [0.01]
    for i in range(1,11):
        kws_leak_percent.append(kws_leak_percent[0]*i*10)   
    
    #kws_leak_percent = [1]
    #BVA_gamma_list = [maximum_gamma]

    Decoding_accuracy = [0]*len(kws_leak_percent)
    Decoding_injection_length = [0]*len(kws_leak_percent)
    Decoding_injection_size = [0]*len(kws_leak_percent)

    BVA_accuracy = [[0]*len(kws_leak_percent) for _ in range(len(BVA_gamma_list))]
    BVA_injection_length = [[0]*len(kws_leak_percent) for _ in range(len(BVA_gamma_list))]
    BVA_injection_size = [[0]*len(kws_leak_percent) for _ in range(len(BVA_gamma_list))]

    BVMA_accuracy = [0]*len(kws_leak_percent)
    BVMA_injection_length = [0]*len(kws_leak_percent)
    BVMA_injection_size = [0]*len(kws_leak_percent)

    SR_1_accuracy = [0]*len(kws_leak_percent)
    SR_1_injection_length = [0]*len(kws_leak_percent)
    SR_1_injection_size = [0]*len(kws_leak_percent)

    SR_linear_accuracy = [0]*len(kws_leak_percent)
    SR_linear_injection_length = [0]*len(kws_leak_percent)
    SR_linear_injection_size = [0]*len(kws_leak_percent)

    Decoding_r_time = 0
    SR_min_r_time = 0
    SR_max_r_time = 0
    BVA_min_r_time = 0
    BVA_max_r_time = 0
    BVMA_r_time = 0
    
    total_loop = len(kws_leak_percent)*len(BVA_gamma_list)
    pbar = tqdm(total=total_loop)
    for ind in range(len(kws_leak_percent)):
        for ind2 in range(len(BVA_gamma_list)):
            Kws_Gama_List = [(kws_leak_percent[ind], BVA_gamma_list[ind2])]*exp_times
            partial_function = partial(multiprocess_worker, chosen_kws, real_size, real_length, offset_of_Decoding, trend_matrix, observed_period, target_period, number_queries_per_period)
            with Pool(processes=exp_times) as pool:
                for result in pool.map(partial_function, Kws_Gama_List):
                    Decoding_accuracy[ind] += (result[0])
                    Decoding_injection_length[ind] += (result[1])
                    Decoding_injection_size[ind] += (result[2])
                    BVA_accuracy[ind2][ind] += (result[3])
                    BVA_injection_length[ind2][ind] += (result[4])
                    BVA_injection_size[ind2][ind] += (result[5])
                    BVMA_accuracy[ind] += (result[6])
                    BVMA_injection_length[ind] += (result[7])
                    BVMA_injection_size[ind] += (result[8])
                    SR_1_accuracy[ind] += (result[9])
                    SR_1_injection_length[ind] += (result[10])
                    SR_1_injection_size[ind] += (result[11])
                    SR_linear_accuracy[ind] += (result[12])
                    SR_linear_injection_length[ind] += (result[13])
                    SR_linear_injection_size[ind] += (result[14])     
                    if ind == len(kws_leak_percent)-1:
                        Decoding_r_time += (result[15])
                        SR_min_r_time += (result[16])
                        SR_max_r_time += (result[17])
                        BVMA_r_time += (result[18])
                        BVA_min_r_time += (result[19])
                        BVA_max_r_time += (result[20])
            BVA_accuracy[ind2][ind] /= exp_times
            BVA_injection_length[ind2][ind] /= exp_times
            BVA_injection_size[ind2][ind] /= exp_times           
            pbar.update(math.ceil((ind*len(BVA_gamma_list)+ind2)/total_loop))
        Decoding_accuracy[ind] /= exp_times
        Decoding_injection_length[ind] /= exp_times
        Decoding_injection_size[ind] /= exp_times
        BVMA_accuracy[ind] /= exp_times
        BVMA_injection_length[ind] /= exp_times
        BVMA_injection_size[ind] /= exp_times
        SR_1_accuracy[ind] /= exp_times
        SR_1_injection_length[ind] /= exp_times
        SR_1_injection_size[ind] /= exp_times
        SR_linear_accuracy[ind] /= exp_times
        SR_linear_injection_length[ind] /= exp_times
        SR_linear_injection_size[ind] /= exp_times
    pbar.close()
    """ save result """
    SaveResult = (kws_leak_percent,  
                Decoding_accuracy, Decoding_injection_length, Decoding_injection_size, 
                BVA_accuracy, BVA_injection_length, BVA_injection_size,
                BVMA_accuracy, BVMA_injection_length, BVMA_injection_size,
                SR_1_accuracy, SR_1_injection_length, SR_1_injection_size, 
                SR_linear_accuracy, SR_linear_injection_length, SR_linear_injection_size)
    with open(utils.RESULT_PATH + '/' + 'AttacksWithKwsLeak{}.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump(SaveResult, f)
        f.close()

    print("Decoding_r_time: {}".format((Decoding_r_time/exp_times)))
    print("SR_min_r_time: {}".format((SR_min_r_time/exp_times)))
    print("SR_max_r_time: {}".format((SR_max_r_time/exp_times)))
    print("BVMA_r_time: {}".format((BVMA_r_time/exp_times)))
    print("BVA_min_r_time: {}".format((BVA_min_r_time/exp_times)))
    print("BVA_max_r_time: {}".format((BVA_max_r_time/exp_times)))
    print()
    """ plot figure """
    plot_figure(dataset_name)

    
