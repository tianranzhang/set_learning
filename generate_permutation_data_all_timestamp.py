import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
#add list of args
parser.add_argument('--perm_time_num', type=int, default= 1)

parser.add_argument('--perm_num', type=int, default= 1)

parser.add_argument('--tolerance', type=int, default= 1000)


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

#Write header
with open("tcn_abnormlabs_baseline/"+"permutation_all_10_label.csv", 'a', newline='') as csvFile:   
        writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
        writer.writerow({'subject_id': 'subject_id', 'seq': 'seq','HF': 'HF'})
csvFile.close()

for i in list(labs_filtered.subject_id.unique()):
    print(i)
    if len(list(labs_filtered.query('subject_id == '+str(i)).hadm_id.unique())) ==1:
        continue;
    curr_subj = labs_filtered.query('subject_id == '+str(i) +'and hadm_id!='+str(list(last_adm_info.query('subject_id == '+str(i)).hadm_id)[0]))
    curr_subj_permutations = []
    all_timestamps = list(set(curr_subj.charttime))
    if args.perm_time_num == -1:
        time_to_permute = all_timestamps
    else:
        unique_charttime = []
        for i in list(curr_subj.charttime):
            if i in unique_charttime:
                continue;
            else:
                unique_charttime = unique_charttime+[i]

        time_to_permute = random.choices(unique_charttime, k = args.perm_time_num)
    #print(time_to_permute)
    charttime_list = []
    for t in list(curr_subj.charttime):
        if t in charttime_list:
            continue;
        charttime_list = charttime_list + [t]
    #charttime_list
    fs=0
    while len(curr_subj_permutations)<args.perm_num:
        if fs>args.tolerance:
            print('ended early, number of timestamps: ',len(charttime_list))
            break;
        fs = fs+1
        curr_subj_permutation = []
        for k in charttime_list:#set(curr_subj.charttime):
            unordered_set = list(curr_subj[curr_subj['charttime'] ==str(k)].event)
            if k in time_to_permute:
                new_seq = list(np.random.permutation(unordered_set))
                curr_subj_permutation = curr_subj_permutation + [new_seq]
            else:
                curr_subj_permutation = curr_subj_permutation + [unordered_set]
        if curr_subj_permutation in curr_subj_permutations:
            continue;
        curr_subj_permutations =curr_subj_permutations+[curr_subj_permutation]            
    with open("tcn_abnormlabs_baseline/"+"permutation_all_10_label.csv", 'a', newline='') as csvFile:   
        writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
        for k in curr_subj_permutations:
                #train_seqs = train_seqs.append({'subject_id': str(i), 'seq': ','.join(i for i in k if i is not None)}, ignore_index=True)
            writer.writerow({'subject_id': str(i), 'seq': ' '.join(str(i).replace('[','').replace(']',''.replace(',','').replace('  ',' ').replace('\'','')) for i in k if i is not None),'HF': list(all_train[all_train['subject_id']== i].HF)[0]})
    csvFile.close()