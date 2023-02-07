import math
import os
import pickle
import random
import utils
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from functools import partial
from tqdm import tqdm
import attacks
def multiprocess_worker(chosen_kws, real_size, real_length, offset_of_Decoding, trend_matrix, observed_period, target_period, number_queries_per_period, gamma):

    begin_time = random.randint(0, len(trend_matrix[0])-observed_period-target_period-1)
    random.shuffle(chosen_kws)
    observed_queries = utils.generate_queries(trend_matrix[:, begin_time:begin_time+observed_period], 'real-world', number_queries_per_period)
    target_queries = utils.generate_queries(trend_matrix[:, begin_time+observed_period:begin_time+observed_period+target_period], 'real-world', number_queries_per_period)

    attack = attacks.Attack(chosen_kws, observed_queries, target_queries, 1.0, trend_matrix, real_size, real_length, offset_of_Decoding)
    if gamma==offset_of_Decoding:
        attack.Decoding_main()
    Decoding_accuracy = attack.accuracy
    Decoding_injection_length = attack.total_inject_length
    Decoding_injection_size = attack.total_inject_size   
    attack.BVA_main(gamma)
    BVA_accuracy = attack.accuracy
    print(BVA_accuracy)
    BVA_injection_length = attack.total_inject_length
    BVA_injection_size = attack.total_inject_size
    return [Decoding_accuracy, Decoding_injection_length, Decoding_injection_size,
            BVA_accuracy, BVA_injection_length, BVA_injection_size]

def plot_figure(dataset_name):
    with open(utils.RESULT_PATH + '/' + 'BVAWithGamma{}.pkl'.format(dataset_name), 'rb') as f:
        (BVA_gamma_list, Decoding_accuracy, Decoding_injection_size, BVA_accuracy, BVA_injection_size) = pickle.load(f)
        f.close()
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], #注意这里是Times，不是Times New Roman
    "font.size":20})
    _, ax = plt.subplots()      
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  
    ax.ticklabel_format(style = 'sci',scilimits=(0,1), axis='x', useMathText=True)
    plt.grid()
    if dataset_name=='Enron':
        wid = 15000
        x_off = BVA_gamma_list.copy()

        ax.bar(x_off[:len(x_off)-2]+[x_off[len(x_off)-1]], BVA_injection_size[:len(BVA_injection_size)-2]+[BVA_injection_size[len(BVA_injection_size)-1]], width=wid, label='BVA-ISize', color = 'green', edgecolor = "white", hatch = '/')
        ax.bar(x_off[len(x_off)-1] + wid , Decoding_injection_size, width=wid, label='Decoding-ISize', color = 'blue', edgecolor = "white", hatch = '\\')
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel('Injection size')
        ax.set_yscale('log')
        ax.tick_params()
        ax.legend(loc = 'center')

        ax2 = ax.twinx()
        ax2.plot(x_off[:len(x_off)-2]+[x_off[len(x_off)-1]], BVA_accuracy[:len(BVA_injection_size)-2]+[BVA_accuracy[len(BVA_accuracy)-1]], 'lightgreen', marker = 'o', markersize=10, markeredgecolor = 'g', markeredgewidth=0.8, label = 'BVA-Rer')
        ax2.scatter(x_off[len(x_off)-1]+wid, Decoding_accuracy, c = 'r', marker = 'x', s=70, label = 'Decoding-Rer')
        ax2.annotate('($\#$W/2, {:.2f})'.format(BVA_accuracy[0]),
        xy=(x_off[0], BVA_accuracy[0]),
        xytext=(x_off[0]-0.01, BVA_accuracy[0]-0.04), color = 'black', 
        arrowprops=dict(arrowstyle="->"))
        ax2.annotate('(offset, {:.2f})'.format(BVA_accuracy[len(BVA_accuracy)-1]),
        xy=(x_off[len(x_off)-1], BVA_accuracy[len(BVA_accuracy)-1]),
        xytext=(x_off[len(x_off)-1]+.025, BVA_accuracy[len(BVA_accuracy)-1]+.015), color = 'black', 
        arrowprops=dict(arrowstyle="->"))
        ax2.set_ylabel('Recovery rate')
        ax2.tick_params()
        ax2.legend(loc = 'upper center', bbox_to_anchor = (0.5,0.9))
    elif dataset_name=='Lucene':
        wid = 45000
        x_off = BVA_gamma_list.copy()

        ax.bar(x_off, BVA_injection_size, width=wid, label='BVA-ISize', color = 'green', edgecolor = "white", hatch = '/')
        ax.bar(x_off[len(x_off)-1] + wid, Decoding_injection_size, width=wid, label='Decoding-ISize', color = 'b', edgecolor = "white", hatch = '\\')
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel('Injection size')
        ax.set_yscale('log')
        ax.tick_params()
        ax.legend(loc = 'center')

        ax2 = ax.twinx()
        ax2.plot(x_off, BVA_accuracy, 'lightgreen', marker = 'o', markersize=10, markeredgecolor = 'g', markeredgewidth=0.8, label = 'BVA-Rer')
        ax2.scatter(x_off[len(x_off)-1]+wid, Decoding_accuracy, c = 'r', marker = 'x', s=70, label = 'Decoding-Rer')
        ax2.annotate('($\#$W/2, {:.2f})'.format(BVA_accuracy[0]),
        xy=(x_off[0], BVA_accuracy[0]),
        xytext=(x_off[0]-0.01, BVA_accuracy[0]-0.037), color = 'black', 
        arrowprops=dict(arrowstyle="->"))
        ax2.annotate('(offset, {:.2f})'.format(BVA_accuracy[len(BVA_accuracy)-1]),
        xy=(x_off[len(x_off)-1], BVA_accuracy[len(BVA_accuracy)-1]),
        xytext=(x_off[len(x_off)-1]+0.025, BVA_accuracy[len(BVA_accuracy)-1]+.015), color = 'black', 
        arrowprops=dict(arrowstyle="->"))
        ax2.set_ylabel('Recovery rate')
        ax2.legend(loc = 'upper center', bbox_to_anchor = (0.5,0.9))
    else:
        wid = 700000
        x_off = BVA_gamma_list.copy()

        ax.bar(x_off, BVA_injection_size, width=wid, label='BVA-ISize', color = 'g', edgecolor = "white", hatch = '/')
        ax.bar(x_off[len(BVA_gamma_list)-1] + wid, Decoding_injection_size, width=wid, label='Decoding-ISize', color = 'b', edgecolor = "white", hatch = '\\')
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel('Injection size')
        ax.set_yscale('log')
        ax.tick_params()
        ax.legend(loc = 'center')

        ax2 = ax.twinx()
        ax2.plot(x_off, BVA_accuracy, 'lightgreen', marker = 'o', markersize=10, markeredgecolor = 'g', markeredgewidth=0.8, label = 'BVA-Rer')
        ax2.scatter(x_off[len(x_off)-1]+wid, Decoding_accuracy, c = 'r', marker = 'x', s=70, label = 'Decoding-Rer')   
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax2.annotate('($\#$W/2, {:.2f})'.format(BVA_accuracy[0]),
        xy=(x_off[0], BVA_accuracy[0]),
        xytext=(x_off[0]-0.01, BVA_accuracy[0]-0.06), color = 'black', 
        arrowprops=dict(arrowstyle="->"))
        ax2.annotate('(offset, {:.2f})'.format(BVA_accuracy[len(BVA_accuracy)-1]),
        xy=(x_off[len(x_off)-1], BVA_accuracy[len(BVA_accuracy)-1]),
        xytext=(x_off[len(x_off)-1]+0.005, BVA_accuracy[len(BVA_accuracy)-1]+0.03), color = 'black', 
        arrowprops=dict(arrowstyle="->"))
        ax2.set_ylabel('Recovery rate')
        ax2.legend(loc = 'upper center', bbox_to_anchor = (0.5,0.9))
    plt.tight_layout()
    plt.savefig(utils.PLOTS_PATH+'/'+'Offset{}.pdf'.format(dataset_name), bbox_inches = 'tight', dpi = 600)
    plt.show()

if __name__ == '__main__':
    
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
    # plot_figure(dataset_name)
    """ read data """
    with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
    chosen_kws = list(kw_dict.keys())
    with open(os.path.join(utils.DATASET_PATH, "{}_wl_v_off.pkl".format(dataset_name.lower())), "rb") as f:
        real_size, real_length, offset_of_Decoding = pickle.load(f)
        f.close()
    print(offset_of_Decoding)
    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), 260, adv_observed_offset)

    """ experiment parameter """
    exp_times = 1
    BVA_gamma_list = []
    minimum_gamma = (int) (len(kw_dict)/2)
    maximum_gamma = offset_of_Decoding
    minimum_gamma += 1
    BVA_gamma_list.append(minimum_gamma)
    while minimum_gamma<maximum_gamma/2:
        minimum_gamma *= 2
        minimum_gamma += 1
        BVA_gamma_list.append(minimum_gamma)
    BVA_gamma_list.append(maximum_gamma)
    Decoding_accuracy = 0
    Decoding_injection_length = 0
    Decoding_injection_size = 0
    BVA_accuracy = [0]*len(BVA_gamma_list)
    BVA_injection_length = [0]*len(BVA_gamma_list)
    BVA_injection_size = [0]*len(BVA_gamma_list)
    
    pbar = tqdm(total=len(BVA_gamma_list))
    for ind2 in range(len(BVA_gamma_list)):    
        partial_function = partial(multiprocess_worker, chosen_kws, real_size, real_length, offset_of_Decoding, trend_matrix, observed_period, target_period, number_queries_per_period)
        with Pool(processes=exp_times) as pool:
            for result in pool.map(partial_function, [BVA_gamma_list[ind2]]*exp_times):
                Decoding_accuracy += result[0]
                Decoding_injection_length += result[1]
                Decoding_injection_size += result[2]
                BVA_accuracy[ind2] += result[3]
                BVA_injection_length[ind2] += result[4]
                BVA_injection_size[ind2] += result[5]
            BVA_accuracy[ind2] /= exp_times
            BVA_injection_length[ind2] /= exp_times
            BVA_injection_size[ind2] /= exp_times
        pbar.update(math.ceil((ind2+1)/len(BVA_gamma_list)))
    pbar.close()
    Decoding_accuracy /= exp_times
    Decoding_injection_length /= exp_times
    Decoding_injection_size /= exp_times
  
    print("Decoding_accuracy:{}".format(Decoding_accuracy))
    print("Decoding_injection_size:{}".format(Decoding_injection_size))
    print("BVA_accuracy:{}".format(BVA_accuracy))
    print("BVA_injection_size:{}".format(BVA_injection_size))
    
    """ save result """
    with open(utils.RESULT_PATH + '/' + 'BVAWithGamma{}.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump((BVA_gamma_list, Decoding_accuracy, Decoding_injection_size, BVA_accuracy, BVA_injection_size), f)
        f.close()

    """ plot figure """
    plot_figure(dataset_name)
    

