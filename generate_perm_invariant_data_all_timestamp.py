import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
#add list of args
parser.add_argument('--perm_time_num', type=int, default= 1)

parser.add_argument('--perm_num', type=int, default= 1)

parser.add_argument('--unique', type=int, default= 0)

parser.add_argument('--ordering', type=str, default= 'None')

parser.add_argument('--tolerance', type=int, default= 1000)

parser.add_argument('--pooling', type = str, default= 'sum')



args = parser.parse_args()




labs_filtered = pd.read_csv('labevents_abnormal.csv')
last_adm_info = pd.read_csv('last_adm_info.csv')
all_train = pd.read_csv('all_train.csv')
all_train = all_train[['subject_id','HF']]
all_train_map = {}

def permutation(lst):
  
    # If lst is empty then there are no permutations 
    if len(lst) == 0: 
        return [] 
  
    # If there is only one element in lst then, only 
    # one permuatation is possible 
    if len(lst) == 1: 
        return [lst] 
  
    # Find the permutations for lst if there are 
    # more than 1 characters 
  
    l = [] # empty list that will store current permutation 
  
    # Iterate the input(lst) and calculate the permutation 
    for i in range(len(lst)):
        
        m = lst[i] 
  
       # Extract lst[i] or m from the list.  remLst is 
       # remaining list 
        remLst = lst[:i] + lst[i+1:] 
  
       # Generating all permutations where m is first 
       # element 
        for p in permutation(remLst): 
            l.append([m] + p) 
    return l



import random
import csv
np.random.seed(100)


import gensim
_model = gensim.models.Word2Vec.load("word2vec.model")

#Write header
with open("tcn_abnormlabs_baseline/"+"pooling_"+str(args.pooling)+".csv", 'a', newline='') as csvFile:   
        writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
        writer.writerow({'subject_id': 'subject_id', 'seq': 'seq','HF': 'HF'})
csvFile.close()

max_timestamp_num = 0
res = pd.DataFrame(columns=('subject_id','seq','HF'))

count = 0 
for i in list(labs_filtered.subject_id.unique()):
    print(i)
    if len(list(labs_filtered.query('subject_id == '+str(i)).hadm_id.unique())) ==1:
        continue;
    curr_subj = labs_filtered.query('subject_id == '+str(i) +'and hadm_id!='+str(list(last_adm_info.query('subject_id == '+str(i)).hadm_id)[0]))
    curr_subj_permutations = []
    all_timestamps = list(set(curr_subj.charttime))
    if len(all_timestamps)>max_timestamp_num:
        max_timestamp_num = len(all_timestamps)
    
    unique_charttime = []
    for e in list(curr_subj.charttime):
        if str(e) in unique_charttime:
            continue;
        else:
            unique_charttime = unique_charttime+[str(e)]

    pooled_sum_sequence = []
    for k in unique_charttime:
        unordered_set = list(curr_subj[curr_subj['charttime'] ==str(k)].event)
        unordered_set = [str(i) for i in unordered_set ]
        unordered_set = [str(i) for i in unordered_set if i not in ['nan']]
        if len(unordered_set)==0:
            continue;

        #print(unordered_set)
        if args.pooling=='sum':
            pooled_sum = np.sum([_model.wv[event] for event in unordered_set], axis = -2)
        elif args.pooling== 'max':
            pooled_sum = np.max([_model.wv[event] for event in unordered_set], axis = -2)
        elif args.pooling== 'mean':
            pooled_sum = np.mean([_model.wv[event] for event in unordered_set], axis = -2)

        
    pooled_sum_sequence = pooled_sum_sequence + [pooled_sum]
    if len(pooled_sum_sequence) < 1095:
            pooled_sum_sequence = [_model.wv['00000']]*(1095-len(pooled_sum_sequence)) + pooled_sum_sequence
    print(len(pooled_sum_sequence))



    with open("tcn_abnormlabs_baseline/"+"pooling_"+str(args.pooling)+".csv", 'a', newline='') as csvFile:   
    #
        writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
            #print(' '.join([str(i) for i in k.split(' ')]))
            #print(all_train[all_train['subject_id']== i].HF)
            
            #print(' '.join([str(i) for i in k.split(' ') if i is not None]))
                #train_seqs = train_seqs.append({'subject_id': str(i), 'seq': ','.join(i for i in k if i is not None)}, ignore_index=True)
        writer.writerow({'subject_id': str(i), 'seq': np.array(pooled_sum_sequence),'HF': list(all_train[all_train['subject_id']== i].HF)[0]})
    csvFile.close()

    res.loc[count] = [str(i), np.array(pooled_sum_sequence),list(all_train[all_train['subject_id']== i].HF)[0]]
    count = count + 1


res.to_pickle("tcn_abnormlabs_baseline/"+"pooling_"+str(args.pooling)+".pkl")
print('pickled object saved')
print("longest sequence length: ", max_timestamp_num)

    #print([str(l) in time_to_permute for l in  unique_charttime])
    #print(unique_charttime)
    #print(len(all_timestamps),' ', len(time_to_permute) , ' ',   len(unique_charttime))
    #charttime_list
    