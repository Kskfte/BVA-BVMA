import os
import pickle
import random
import matplotlib
import matplotlib.pyplot as plt
import utils

def get_kws_from_known_doc(chosen_kws, divide_doc_percentage, doc):
    KNOWN_LIST = []
    for i in divide_doc_percentage:
        known_partial_doc = doc[:(int) (len(doc)*i)]
        known_kws = {}
        for kws in known_partial_doc:
            for kw in kws:
                if kw in chosen_kws.keys():
                    known_kws[kw] = 0
        KNOWN_LIST.append(len(known_kws))
    return KNOWN_LIST

def plot_figure(divide_doc_percentage, known_kws_from_enron_dataset, known_kws_from_lucene_dataset):
    #textsize = 16
    #font = {'family':'Times New Roman', 'size': 16}
    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": True,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], #注意这里是Times，不是Times New Roman
    "font.size":20})
    plt.plot(divide_doc_percentage, known_kws_from_enron_dataset, 'lightsalmon', marker = 'o', markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = 'Enron File')
    plt.plot(divide_doc_percentage, known_kws_from_lucene_dataset, 'lightgreen', marker = 's', markeredgecolor = 'green', markersize = 10, markeredgewidth=0.8, label = 'Lucene File')
    plt.xlabel('File leakage percentage')
    plt.ylabel('No. of Kws in Enron')
    plt.tick_params()
    plt.grid()
    plt.legend()
    plt.savefig(utils.PLOTS_PATH + '/' + 'KWDistribution.pdf', bbox_inches = 'tight', dpi = 600)#.pdf 
    plt.show()

if __name__=='__main__':

    if not os.path.exists(utils.RESULT_PATH):
        os.makedirs(utils.RESULT_PATH)
    if not os.path.exists(utils.PLOTS_PATH):
        os.makedirs(utils.PLOTS_PATH)
    """ dataset """
    with open(os.path.join(utils.DATASET_PATH,"enron_doc.pkl"), "rb") as f:
        enron_doc = pickle.load(f)
        f.close()
    with open(os.path.join(utils.DATASET_PATH,"enron_kws_dict.pkl"), "rb") as f:
        enron_kw_dict = pickle.load(f)
        f.close()
    with open(os.path.join(utils.DATASET_PATH,"lucene_doc.pkl"), "rb") as f:
        lucene_doc = pickle.load(f)
        f.close()  
    enron_chosen_kws = enron_kw_dict
    random.shuffle(enron_doc)
    random.shuffle(lucene_doc)
    divide_doc_percentage = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
    known_kws_from_enron_dataset = [0]*len(divide_doc_percentage)
    known_kws_from_lucene_dataset = [0]*len(divide_doc_percentage)
    known_kws_from_enron_dataset = get_kws_from_known_doc(enron_kw_dict, divide_doc_percentage, enron_doc)
    known_kws_from_lucene_dataset = get_kws_from_known_doc(enron_kw_dict, divide_doc_percentage, lucene_doc)
    print(known_kws_from_enron_dataset)
    print(known_kws_from_lucene_dataset)
 
    plot_figure(divide_doc_percentage, known_kws_from_enron_dataset, known_kws_from_lucene_dataset)
