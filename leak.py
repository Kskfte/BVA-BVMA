import os
import pickle
import utils
import sys
def get_kws_size_and_length(doc, chosen_kws, real_size, real_length):
    """
    get all keywords response size and response length.
    """
    kw_to_id = {}
    for kw_id in range(len(chosen_kws)):
        kw_to_id[chosen_kws[kw_id]] = kw_id
    for _, cli_doc_kws in enumerate(doc):
        flag = [False]*len(chosen_kws) 
        for doc_kws in cli_doc_kws:
            if doc_kws in kw_to_id.keys() and not flag[kw_to_id[doc_kws]]:
                real_size[kw_to_id[doc_kws]] += len(cli_doc_kws)
                real_length[kw_to_id[doc_kws]] += 1
                flag[kw_to_id[doc_kws]] = True

if __name__=='__main__':
    d_id = input("input evaluation dataset: 1. Enron 2. Lucene 3.WikiPedia ")
    dataset_name = ''
    if d_id=='1':
        dataset_name = 'Enron'
    elif d_id=='2':
        dataset_name = 'Lucene'  
    elif d_id=='3':
        dataset_name = 'Wiki'
    else:
        raise ValueError('No Selected Dataset!!!')

    if dataset_name != 'Wiki':
        with open(os.path.join(utils.DATASET_PATH,"{}_doc.pkl".format(dataset_name.lower())), "rb") as f:
            doc = pickle.load(f)
            f.close()
    else: 
        doc = []
        for t in range(6):
            with open(os.path.join(utils.DATASET_PATH,"{}_doc_{}.pkl".format(dataset_name.lower(), t)), "rb") as f:
                doc += pickle.load(f)
                f.close()
    
    with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
    chosen_kws = list(kw_dict.keys())
    real_size, real_length = utils.get_kws_size_and_length(doc, chosen_kws)
    offset_of_Decoding = utils.compute_decoding_offset(real_size, sys.maxsize)
    with open(os.path.join(utils.DATASET_PATH, "{}_wl_v_off.pkl".format(dataset_name, dataset_name.lower())), "wb") as f:
        pickle.dump((real_size, real_length, offset_of_Decoding), f)
        f.close()

    