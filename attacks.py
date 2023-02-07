import time
import math
import numpy as np
import random
from collections import Counter

class Attack: 
    def __init__(self, chosen_kws, observed_queries, target_queries, kws_leak_percent, trend_matrix_norm, real_size, real_length, offset_of_Decoding):
        self.real_tag = {}
        self.recover_tag = {}

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
  
        self.real_size, self.real_length = real_size, real_length

        self.BVA_min_rer_time = 0
        self.BVA_max_rer_time = 0
        self.SR_min_rer_time = 0
        self.SR_max_rer_time = 0
        self.BVMA_rer_time = 0
        self.Decoding_rer_time = 0


        """
        baseline phase
        """
        self.observed_size, self.max_observed_size, self.observed_length = self.get_baseline_observed_size_and_length(self.real_size, self.real_length)
        """
        get offset of Decoding
        """
        self.offset = offset_of_Decoding 

    def get_baseline_observed_size_and_length(self, real_size, real_length):
        """
        observe size and length in baseline phase
        """
        observed_size = {}
        observed_length = {}
        max_observed_size = 0
        for i_week in self.observed_queries:
            for query in i_week:
                observed_size[query] = real_size[query]
                observed_length[query] = real_length[query]
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
        injection
        """
        s = time.time()
        real_size_after_injection = self.real_size.copy()
        self.Decoding_inject(real_size_after_injection)
        e = time.time()
        self.inject_time = e - s
        """
        recovery
        """
        observed_size_in_baseline = self.observed_size.copy()
        s = time.time()
        self.Decoding_recover(observed_size_in_baseline, real_size_after_injection)
        e = time.time()
        self.Decoding_rer_time = e - s
        #print("DecodingRecoeryTime:{}".format(self.Decoding_rer_time))
        
        self.total_inject_length = (int) (len(self.chosen_kws)*self.kws_leak_percent) - 1
        self.accuracy = self.recover_queries_num/self.total_queries_num
    def Decoding_recover(self, observed_size_in_baseline, real_size_after_injection):
        self.real_tag = {}
        self.recover_tag = {}

        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                for kw_id in observed_size_in_baseline.keys():
                    if query in real_size_after_injection.keys(): 
                        if (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) % self.offset == 0:
                            self.recover_tag[query] = (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) / self.offset
                            break
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def Decoding_inject(self, real_size_after_injection):
        """
        injection: injection size and real_size_after_injection
        """
        for kw_id in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
            self.total_inject_size += kw_id*self.offset 
            real_size_after_injection[kw_id] += kw_id*self.offset

    def BVA_main(self, gamma):
        self.total_queries_num = 0
        self.recover_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0
        self.gamma = gamma
        """
        inject
        """
        s = time.time()
        real_size_after_injection = self.real_size.copy()
        self.BVA_inject(real_size_after_injection)
        e = time.time()
        self.inject_time = e - s
        """
        recovery
        """
        observed_size_in_baseline = self.observed_size.copy()
        s = time.time()
        self.BVA_recover(observed_size_in_baseline, real_size_after_injection)
        e = time.time()
        if self.gamma==(int)(len(self.chosen_kws)/2):
            self.BVA_min_rer_time = e-s
        elif self.gamma==(int)(self.offset/4):
            self.BVA_max_rer_time = e-s
        # print("BVARecoeryTime:{}".format(self.time))
        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        self.total_inject_length = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        self.accuracy = self.recover_queries_num/self.total_queries_num
    def BVA_recover(self, observed_size_in_baseline, real_size_after_injection):
        self.real_tag = {}
        self.recover_tag = {}

        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                for kw_id in observed_size_in_baseline.keys():
                    if query in real_size_after_injection.keys(): 
                        if (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) % self.gamma == 0:
                            self.recover_tag[query] = (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) / self.gamma
                            break
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def BVA_inject(self, real_size_after_injection):
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
        for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
            for num_ind in range(num_injection_doc):
                if ((kws_ind >> num_ind) & 1) == 1 :
                    real_size_after_injection[kws_ind] += size_each_doc[num_ind]       

    def BVMA_NoSP_main(self):
        self.total_queries_num = 0
        self.recover_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0

        """
        injection
        """
        s = time.time()
        real_size_after_injection = self.real_size.copy()
        real_length_after_injection = self.real_length.copy()
        self.BVMA_NoSP_inject(real_size_after_injection, real_length_after_injection)
        e = time.time()
        self.inject_time = e - s
        """
        recovery
        """
        observed_size_in_baseline = self.observed_size.copy()
        observed_length_in_baseline = self.observed_length.copy()
        #s = time.time()
        self.BVMA_NoSP_recover(observed_size_in_baseline, observed_length_in_baseline, real_size_after_injection, real_length_after_injection)
        #e = time.time()
        #self.time = e - s
        # print("BVMATIME:{}".format(self.time))
        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        self.total_inject_length = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        self.accuracy = self.recover_queries_num/self.total_queries_num
    def BVMA_NoSP_recover(self, observed_size_in_baseline, observed_length_in_baseline, real_size_after_injection, real_length_after_injection):
        self.real_tag = {}
        self.recover_tag = {}
            
        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                self.recover_tag[query] = -1
                #findflag = False
                for kw_id in observed_size_in_baseline.keys():
                    if query in real_size_after_injection.keys() and real_size_after_injection[query] - observed_size_in_baseline[kw_id] >= 0 and (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) - (len(self.chosen_kws)/2)*math.log2(len(self.chosen_kws)) < len(self.chosen_kws):                           
                        diffReBa = (int) ((real_size_after_injection[query] - observed_size_in_baseline[kw_id]))
                        t_counteae = 0
                        num_tF = 0
                        while(diffReBa>=0):
                            diffReBa -= (int) (len(self.chosen_kws)/2)
                            if diffReBa<0:
                                break
                            t_counteae += 1
                            if diffReBa<len(self.chosen_kws):
                                re_kw_id = diffReBa
                                tmp_kw_id = re_kw_id #recovery kw_id
                                num_tF = 0 # theoretical injection length
                                while tmp_kw_id!=0:
                                    if tmp_kw_id&1==1:
                                        num_tF += 1
                                    tmp_kw_id >>= 1
                                if t_counteae!=num_tF:
                                    continue
                                if real_length_after_injection[query] - observed_length_in_baseline[kw_id] == num_tF:
                                    self.recover_tag[query] = re_kw_id
                                    #findflag = True
                                    #break
                        #if findflag:
                        #    break
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def BVMA_NoSP_inject(self, real_size_after_injection, real_length_after_injection):       
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
        for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
            for num_ind in range(num_injection_doc):
                if ((kws_ind >> num_ind) & 1) == 1 :
                    real_size_after_injection[kws_ind] += size_each_doc[num_ind]
                    real_length_after_injection[kws_ind] += 1

        return real_size_after_injection, real_length_after_injection

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
        injection
        """
        s = time.time()
        real_size_after_injection = self.real_size.copy()
        real_length_after_injection = self.real_length.copy()
        self.BVMA_SP_inject(real_size_after_injection, real_length_after_injection)
        e = time.time()
        self.inject_time = e - s
        """
        recovery
        """
        observed_size_in_baseline = self.observed_size.copy()
        observed_length_in_baseline = self.observed_length.copy()
        s = time.time()
        self.time = self.BVMA_SP_recover('real-world', baseline_trend_matrix, observed_size_in_baseline, observed_length_in_baseline, real_size_after_injection, real_length_after_injection)
        e = time.time()
        self.BVMA_rer_time = e - s
        # print("BVMARecoeryTime:{}".format(self.time))
        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        self.total_inject_length = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        self.accuracy = self.recover_queries_num/self.total_queries_num
    def BVMA_SP_recover(self, query_type, baseline_trend_matrix, observed_size_in_baseline, observed_length_in_baseline, real_size_after_injection, real_length_after_injection):
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
                for kw_id in observed_size_in_baseline.keys():
                    if query in real_size_after_injection.keys() and real_size_after_injection[query] - observed_size_in_baseline[kw_id] >= 0 and (real_size_after_injection[query] - observed_size_in_baseline[kw_id]) - (len(self.chosen_kws)/2)*math.log2(len(self.chosen_kws)) < len(self.chosen_kws):                           
                        diffReBa = (int) ((real_size_after_injection[query] - observed_size_in_baseline[kw_id])/1)
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
                                if real_length_after_injection[query] - observed_length_in_baseline[kw_id] == num_tF:
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
    def BVMA_SP_inject(self, real_size_after_injection, real_length_after_injection):       
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
        for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
            for num_ind in range(num_injection_doc):
                if ((kws_ind >> num_ind) & 1) == 1 :
                    real_size_after_injection[kws_ind] += size_each_doc[num_ind]
                    real_length_after_injection[kws_ind] += 1

        return real_size_after_injection, real_length_after_injection

    def SR_main(self, m):
        self.total_queries_num = 0
        self.recover_queries_num = 0
        self.total_inject_length = 0
        self.total_inject_size = 0
        self.accuracy = 0
        self.m = m
        """
        inject
        """
        #s = time.time()
        real_length_after_injection = self.real_length.copy()
        self.SR_inject(real_length_after_injection)
        #e = time.time()
        #self.inject_time = e - s
        """
        recovery
        """
        s = time.time()
        self.SR_recover(real_length_after_injection)
        e = time.time()
        if self.m==1:
            self.SR_min_rer_time = e-s
        elif self.m==len(self.chosen_kws):
            self.SR_max_rer_time = e-s
        # print("SR-{}RecoeryTime:{}".format(self.m, self.time))

        self.accuracy = self.recover_queries_num/self.total_queries_num
    def SR_recover(self, real_length_after_injection):
        self.real_tag = {}
        self.recover_tag = {}
        for i_week in self.target_queries:
            for query in i_week:
                self.real_tag[query] = query
                if real_length_after_injection[query]<self.m:
                    self.recover_tag[query] = random.randint(0, len(self.chosen_kws)-1)
                else:
                    self.recover_tag[query] = math.floor(real_length_after_injection[query]/self.m)-1
                #for kw_id in observed_length_in_baseline.keys():
                #    if query in real_length_after_injection.keys(): 
                #        if (real_length_after_injection[query] - observed_length_in_baseline[kw_id]) % self.m == 0:
                #            self.recover_tag[query] = (real_length_after_injection[query] - observed_length_in_baseline[kw_id]) / self.m - 1
                #            break
                if self.recover_tag[query] == self.real_tag[query]:
                    self.recover_queries_num += 1
                self.total_queries_num += 1
    def SR_inject(self, real_length_after_injection):
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

        self.total_inject_size = self.m*self.kws_leak_percent*known_kws_num*(known_kws_num+1)/2
        

        for kws_ind in range(known_kws_num):
             real_length_after_injection[kws_ind] += (kws_ind+1)*self.m     
