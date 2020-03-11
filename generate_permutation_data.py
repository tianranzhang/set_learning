import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
#add list of args
parser.add_argument('--perm_time_num', type=int, default= 1)

parser.add_argument('--perm_num', type=int, default= 1)



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
with open("tcn_abnormlabs_baseline/"+"permutation_1_2_label.csv", 'a', newline='') as csvFile:   
        writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
        writer.writerow({'subject_id': 'subject_id', 'seq': 'seq','HF': 'HF'})
csvFile.close()

for i in list(labs_filtered.subject_id.unique()):
    print(i)
    if len(list(labs_filtered.query('subject_id == '+str(i)).hadm_id.unique())) ==1:
        continue;
    curr_subj = labs_filtered.query('subject_id == '+str(i) +'and hadm_id!='+str(list(last_adm_info.query('subject_id == '+str(i)).hadm_id)[0]))
    curr_subj_permutations = [list()]
    time_to_permutate = random.choice(list(curr_subj.charttime))
    print(time_to_permutate)
    charttime_list = []
    for t in list(curr_subj.charttime):
        if t in charttime_list:
            continue;
        charttime_list = charttime_list + [t]
    #charttime_list
    for k in charttime_list:#set(curr_subj.charttime):
    #print(k)
        unordered_set = list(curr_subj[curr_subj['charttime'] ==str(k)].event)
        if k == time_to_permutate:
            perm_list = []
            if len(unordered_set)<3:
                perm_list = permutation(unordered_set)
            else:
                while len(perm_list)<2:
                    new_seq = list(np.random.permutation(unordered_set))
                    if new_seq in perm_list:
                        continue;
                    perm_list =perm_list+[new_seq]
            
            for j in curr_subj_permutations:
                curr_subj_permutations = [j+l for l in perm_list]
        else:
            curr_subj_permutations = [i + unordered_set for i in curr_subj_permutations]
                
    with open("tcn_abnormlabs_baseline/"+"permutation_1_2_label.csv", 'a', newline='') as csvFile:   
        writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
        for k in curr_subj_permutations:
                #train_seqs = train_seqs.append({'subject_id': str(i), 'seq': ','.join(i for i in k if i is not None)}, ignore_index=True)
            writer.writerow({'subject_id': str(i), 'seq': ' '.join(str(i) for i in k if i is not None),'HF': list(all_train[all_train['subject_id']== i].HF)[0]})
    csvFile.close()
