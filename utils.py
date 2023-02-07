import random
import numpy as np
import math

DATASET_PATH = 'Datasets'
RESULT_PATH = 'Results'
PLOTS_PATH = 'Plots'

def compute_decoding_offset(observed_response_size, threshold):          
    """compute offset of decoding"""
    divided_list = {}
    for i in observed_response_size.keys():
        for j in observed_response_size.keys():
            temp_size_minus = abs(observed_response_size[i] - observed_response_size[j])
            if temp_size_minus!=0:
                divided_list[temp_size_minus] = 0

    
    #with open('divide_list.pkl', 'rb') as f:
    #    divided_list = pickle.load(f)
    #    f.close()
    #print(len(divided_list))
    offset = 0
    off_bound = 10**7
    for injection_offset in range(2, off_bound):
        #print(injection_offset)
        if injection_offset%10000==0:
            print(injection_offset)
        flag = True
        satisfied_size_minus = 0
        for divisor in divided_list.keys():
            if divisor % injection_offset == 0:
                flag = False
                break
            else:
                satisfied_size_minus += 1
                if satisfied_size_minus>=threshold:
                    break     
        if flag:
            offset = injection_offset
            break       
    if offset == 0:
        print("non adaptive offset!!!")
        offset = off_bound
    print("offset: {}".format(offset))
    return offset

def generate_keyword_trend_matrix(kw_dict, n_kw, n_weeks, offset):
    """
    generate keywords and their queries' trend matrix(row: kws; colume: weeks)
    @ param: kw_dict, n_kw, n_weeks, adv. known weeks offset
    @ return: chosen_kws, trend_matrix, offset_trend_matrix
    """
    
    # n_kw number of keywords
    keywords = list(kw_dict.keys())
    permutation = np.random.permutation(len(keywords))
    #chosen_kws = [keywords[idx] for idx in range(n_kw)]
    chosen_kws = [keywords[idx] for idx in permutation[: n_kw]]
    #print(chosen_kws)
    # n_weeks trend
    trend_matrix_norm = np.array([[float(kw_dict[kw]['trend'][i]) for i in range(len(kw_dict[kw]['trend']))] for kw in chosen_kws])
    #print((trend_matrix_norm[:, 1]))
    for i_col in range(trend_matrix_norm.shape[1]):
        if sum(trend_matrix_norm[:, i_col]) == 0:
            print("The {d}th column of the trend matrix adds up to zero, making it uniform!")
            trend_matrix_norm[:, i_col] = 1 / n_kw
        else:
            trend_matrix_norm[:, i_col] =  trend_matrix_norm[:, i_col] / (float) (sum(trend_matrix_norm[:, i_col]))
    #print((trend_matrix_norm[:, 1]))
    #print(max(trend_matrix_norm[:, 1]))
    #print(trend_matrix_norm[:, 1].index(max(trend_matrix_norm[:, 1])))
    return chosen_kws, trend_matrix_norm[:, -n_weeks:], trend_matrix_norm[:, -n_weeks:] if offset ==0 else trend_matrix_norm[:, -offset-n_weeks: -offset]


def generate_queries(trend_matrix_norm, q_mode, n_qr):
    """
    generate queries from different query modes
    @ param: trend_matrix, q_mode = ['trend', 'uniform']
    @ return: queries matrix(each week)(each kw_id in the chosen_kws)
    """
    queries = []
    n_kw, n_weeks = trend_matrix_norm.shape
    if q_mode == 'real-world':
        for i_week in range(n_weeks):
            #n_qr_i_week = np.random.poisson(n_qr)
            n_qr_i_week = n_qr
            queries_i_week = list(np.random.choice(list(range(n_kw)), n_qr_i_week, p = trend_matrix_norm[:, i_week]))
            queries.append(queries_i_week)

    elif q_mode == 'uniform':
        for i_week in range(n_weeks):
            queries_i_week = list(np.random.choice(list(range(n_kw)), n_qr))
            queries.append(queries_i_week)
    else:
        raise ValueError("Query params not recognized")        
    return queries

def get_kws_id(chosen_kws):
    """
    get all keywords response size and response length.
    """
    kw_to_id = {}
    for kw_id in range(len(chosen_kws)):
        kw_to_id[chosen_kws[kw_id]] = kw_id
    return kw_to_id

def get_kws_size_and_length(doc, chosen_kws):
    """
    get all keywords response size and response length.
    """
    real_size = {}
    real_length = {}
    kw_to_id = {}
    for kw_id in range(len(chosen_kws)):
        real_size[kw_id] = 0
        real_length[kw_id] = 0
        kw_to_id[chosen_kws[kw_id]] = kw_id
    for _, cli_doc_kws in enumerate(doc):
        flag = [False]*len(chosen_kws) 
        for doc_kws in cli_doc_kws:
            if doc_kws in kw_to_id.keys() and not flag[kw_to_id[doc_kws]]:
                real_size[kw_to_id[doc_kws]] += len(cli_doc_kws)
                real_length[kw_to_id[doc_kws]] += 1
                flag[kw_to_id[doc_kws]] = True
    return real_size, real_length


def get_file_size(doc):
    """
    get all keywords response size and response length.
    """
    max_file_size = 0
    min_file_size = 10**9
    total_size = 0
    l = len(doc)
    for _, cli_doc_kws in enumerate(doc):
        
        total_size += len(cli_doc_kws)
        if max_file_size<len(cli_doc_kws):
            max_file_size = len(cli_doc_kws)
        if min_file_size>len(cli_doc_kws):
            min_file_size = len(cli_doc_kws)
    avg_file_size = (int)(total_size/l)
    return min_file_size, max_file_size, avg_file_size

def get_min_max_SEAL_exponent(kws_length, base, Group):
    min_exponent = 50
    max_exponent = -1
    for k in kws_length.keys():
        m = base
        tmp_exp = 1
        while m!=kws_length[k]:
            m *= base
            tmp_exp += 1
        if tmp_exp < min_exponent:
            min_exponent = tmp_exp
        if tmp_exp > max_exponent:
            max_exponent = tmp_exp
    return min_exponent, max_exponent
        

def get_kws_size_and_length_after_padding(doc, chosen_kws, x):
    """
    get all keywords response size and response length.
    """
    real_size = {}
    real_length = {}
    kw_to_id = {}
    max_file_size = 0
    min_file_size = 10**9
    for kw_id in range(len(chosen_kws)):
        real_size[kw_id] = 0
        real_length[kw_id] = 0
        kw_to_id[chosen_kws[kw_id]] = kw_id
    for _, cli_doc_kws in enumerate(doc):
        if max_file_size<len(cli_doc_kws):
            max_file_size = len(cli_doc_kws)
        if min_file_size>len(cli_doc_kws):
            min_file_size = len(cli_doc_kws)
        flag = [False]*len(chosen_kws) 
        for doc_kws in cli_doc_kws:
            if doc_kws in kw_to_id.keys() and not flag[kw_to_id[doc_kws]]:
                real_size[kw_to_id[doc_kws]] += len(cli_doc_kws)
                real_length[kw_to_id[doc_kws]] += 1
                flag[kw_to_id[doc_kws]] = True
    """ SEAL strategy """
    #print(min_file_size)
    #print(max_file_size)
    if x==0:
        return real_size, real_length
    for k in real_length.keys():
        m = x
        while real_length[k]>m:
            m *= x  
        for _ in range(m - real_length[k]):
            real_size[k] += random.randint(min_file_size, max_file_size)
        real_length[k] = m
    return real_size, real_length


def get_kws_size_and_length_after_seal(doc, chosen_kws, x, B):
    """
    get all keywords response size and response length.
    """
    
    size_without_countermeasure = {}
    length_without_countermeasure = {}
    size_after_SEAL = {}
    length_after_SEAL = {}
    
    kw_to_id = {}
    for kw_id in range(len(chosen_kws)):
        size_without_countermeasure[kw_id] = 0
        length_without_countermeasure[kw_id] = 0
        size_after_SEAL[kw_id] = 0
        length_after_SEAL[kw_id] = 0
        kw_to_id[chosen_kws[kw_id]] = kw_id
    for _, cli_doc_kws in enumerate(doc):
        flag = [False]*len(chosen_kws) 
        for doc_kws in cli_doc_kws:
            if doc_kws in kw_to_id.keys() and not flag[kw_to_id[doc_kws]]:
                length_without_countermeasure[kw_to_id[doc_kws]] += 1
                size_without_countermeasure[kw_to_id[doc_kws]] += len(cli_doc_kws)

                number_shard = math.ceil(len(cli_doc_kws)/B)
                length_after_SEAL[kw_to_id[doc_kws]] += number_shard
                size_after_SEAL[kw_to_id[doc_kws]] += number_shard*B
                flag[kw_to_id[doc_kws]] = True
    """ SEAL strategy """
    if x==0:
        return size_without_countermeasure, length_without_countermeasure
    for k in length_after_SEAL.keys():
        m = x
        while length_after_SEAL[k]>m:
            m *= x  
        size_after_SEAL[k] += (m - length_after_SEAL[k])*B
        length_after_SEAL[k] = m
    #print(length_after_SEAL)
    #print(size_after_SEAL)

    return size_after_SEAL, length_after_SEAL

def get_kws_size_and_length_after_obfuscation(doc, chosen_kws, stratey):
    """
    get all keywords response size and response length.
    CLRZ strategy
    """
    p = 0.88703
    q = 0.04416
    real_size = {}
    real_length = {}
    kw_to_id = {}
    total_length = 0
    total_size = 0
    for kw_id in range(len(chosen_kws)):
        real_size[kw_id] = 0
        real_length[kw_id] = 0
        kw_to_id[chosen_kws[kw_id]] = kw_id
    if stratey=='Expected':
        for _, cli_doc_kws in enumerate(doc):
            total_length += 1
            total_size += len(cli_doc_kws)
            flag = [False]*len(chosen_kws) 
            for doc_kws in cli_doc_kws:
                if doc_kws in kw_to_id.keys() and not flag[kw_to_id[doc_kws]]:
                    flag[kw_to_id[doc_kws]] = True
                    real_size[kw_to_id[doc_kws]] += len(cli_doc_kws)
                    real_length[kw_to_id[doc_kws]] += 1
                    
        """ CLRZ strategy """
        for k in real_length.keys():
            real_length[k] = p*real_length[k]+q*(total_length-real_length[k])
            real_size[k] = p*real_size[k]+q*(total_size-real_size[k])
    else:
        count = 0
        for doc_kws in kw_to_id.keys():
            count+=1
            print(count)
            for _, cli_doc_kws in enumerate(doc):
                if doc_kws in cli_doc_kws:
                    if random.random()<p:
                        real_size[kw_to_id[doc_kws]] += len(cli_doc_kws)
                        real_length[kw_to_id[doc_kws]] += 1
                else:
                    if random.random()<q:
                        real_size[kw_to_id[doc_kws]] += len(cli_doc_kws)
                        real_length[kw_to_id[doc_kws]] += 1
    return real_size, real_length
