# Code for: ''High Recovery with Fewer Injections: Practical Binary Volumetric Injection Attacks against Dynamic Searchable Encryption''

## Summary of files
The basic files in our code are:
* ``leak.py``: with the input of ''enron_doc'' and ''enron_kws_list'', this file creates a ''enron_wl_v_off.pkl'' for Enron (also can be used for Lucene and Wikipedia) contains response length pattern and response size pattern of all keywords, and  the ''offset'' of ''Decoding'' attack. Actually, we have provided the file ''enron_wl_v_off.pkl'' after running locally. Therefore, one can run ``fig_*.py`` directly to view the results quickly.
* ``attacks.py``: implements the different attacks we evaluate in our paper, mainly contains  ''Decoding'', ''Search'', ''BVA'', and ''BVMA''.
* ``utils.py``: provides access to different functions that we use in the code (e.g., generate keywords trend matrix and simulate real queries, compute ''offset'' and volume pattern).

Other files ``fig_*.py`` is used to runn attacks for different evaluations shown in paper. Running these files to save the results in pickle files, and meanwhile plot the figure in our paper.

## Datasets
The processed datasets are under the ''Datasets'' folder for the three Enron, Lucene and Wikipedia. Each dataset contains three pickle files: ''doc.pkl'', ''kws_dict.pkl'', ''wl_v_off.pkl''.
1) ``doc.pkl``: is a list of lists (documents). Each document is a list of the words (strings, containing keywords and other words) associated to that document.
2) ``kws_dict.pkl``: is a dictionary whose keys are the keywords (strings). The values are dictionaries with two keys:
 
    a) ``'count'``, that maps to the number of times that keyword appears in the dataset.
 
    b) ``'trend'``, that maps to a numpy array with the trends of this keyword.
    
3) ``wl_v_off.pkl``: is a dictionary containing the volume information of all keywords and offset of ''Decoding'' after running ''leak.py''. 
 
## Instructions
One can directly run ``fig_*.py`` or ``table_*.py`` to save and plot the results.
