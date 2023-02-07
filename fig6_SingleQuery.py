import os
import math
import utils
import pickle
import matplotlib.pyplot as plt
from math import log2

class SingleQuery: 
    def __init__(self, chosen_kws):

        self.chosen_kws = chosen_kws
    def BVA(self, gamma):
        """
        return injection size and round
        """
        ISize = gamma*len(self.chosen_kws)
        Round = 1
        return ISize, Round

    def BVMA(self):
        """
        return injection size and round
        """
        ISize = len(self.chosen_kws) +  (int) (len(self.chosen_kws)*math.ceil(log2(len(self.chosen_kws)))/2)
        Round = 1
        return ISize, Round

    def Multiple_round(self, k):
        """
        return injection size and round
        """
        ISize = k*len(self.chosen_kws)
        Round = math.ceil(log2(len(self.chosen_kws))/log2(k))
        return ISize, Round

    def Search(self):
        """
        return injection size and round
        """
        ISize = len(self.chosen_kws)
        Round = math.ceil(log2(len(self.chosen_kws)))
        return ISize, Round

if __name__=='__main__':
    
    if not os.path.exists(utils.RESULT_PATH):
        os.makedirs(utils.RESULT_PATH)
    if not os.path.exists(utils.PLOTS_PATH):
        os.makedirs(utils.PLOTS_PATH)
    """ choose dataset """
    d_id = input("input evaluation dataset: 1. Enron 2. Lucene 3.WikiPedia ")
    dataset_name = ''
    print(d_id)
    if d_id=='1':
        dataset_name = 'Enron'
    elif d_id=='2':
        dataset_name = 'Lucene'  
    elif d_id=='3':
        dataset_name = 'Wiki'
    else:
        raise ValueError('No Selected Dataset!!!')

    """ experiment """
    with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close() 
    chosen_kws = list(kw_dict.keys())
    SG = SingleQuery(chosen_kws)
    QueryList = list(range(len(chosen_kws)))
    BVAISizeList = []
    BVMAISizeList = []
    MGISizeList = []
    SISizeList = []
    BVARoundList = []
    BVMARoundList = []
    MGRoundList = []
    SRoundList = []
    for i in range(1, len(QueryList)+1):
        IS_R = SG.BVA((int)(len(chosen_kws)/2))
        BVAISizeList.append(IS_R[0]/i)
        BVARoundList.append(IS_R[1])
        IS_R = SG.BVMA()
        BVMAISizeList.append(IS_R[0]/i)
        BVMARoundList.append(IS_R[1])
        IS_R = SG.Multiple_round(2)
        MGISizeList.append(IS_R[0])
        MGRoundList.append(IS_R[1])
        IS_R = SG.Search()
        SISizeList.append(IS_R[0])
        SRoundList.append(IS_R[1])

    plt.rcParams.update({
    "legend.fancybox": False,
    "legend.frameon": False,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"], #注意这里是Times，不是Times New Roman
    "font.size":20})
    plt.figure()         
    plt.plot(QueryList, MGISizeList, linestyle = '-', color = 'y', label='Multiple-round')
    plt.plot(QueryList, SISizeList,  linestyle = '--', color = 'r',label='Search')
    plt.plot(QueryList, BVAISizeList, linestyle = '-.', color = 'g',label='BVA')
    plt.plot(QueryList, BVMAISizeList, linestyle = ':', color = 'b',label='BVMA')
    plt.legend(frameon = True)
    plt.grid(axis='x')
    plt.yscale('log')  
    plt.xlabel('No. of queries') 
    plt.ylabel('Average injection size per query') 
    plt.tight_layout()
    plt.savefig(utils.PLOTS_PATH + '/' + 'SingleQuery{}.pdf'.format(dataset_name), bbox_inches = 'tight', dpi = 600)
    plt.show()