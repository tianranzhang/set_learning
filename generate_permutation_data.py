import pandas as pd
import numpy as np
labs_filtered = pd.read_csv('labevents_abnormal.csv')
last_adm_info = pd.read_csv('last_adm_info.csv')
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
with open("tcn_abnormlabs_baseline/"+"permutation_1_10.csv", 'a', newline='') as csvFile:   
        writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq'])
        writer.writerow({'subject_id': 'subject_id', 'seq': 'seq'})
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
            for cnt in np.arange(10):
                new_seq = list(np.random.permutation(unordered_set))
                #if new_seq in perm_list:
                    #continue;
                perm_list =perm_list+[new_seq]
            #if len(perm_list)>10:
                #perm_list = random.choices(perm_list,k=10)
            for j in curr_subj_permutations:
                curr_subj_permutations = [j+l for l in perm_list]
        else:
            curr_subj_permutations = [i + unordered_set for i in curr_subj_permutations]
                
    with open("tcn_abnormlabs_baseline/"+"permutation_1_10.csv", 'a', newline='') as csvFile:   
        writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq'])
        for k in curr_subj_permutations:
                #train_seqs = train_seqs.append({'subject_id': str(i), 'seq': ','.join(i for i in k if i is not None)}, ignore_index=True)
            writer.writerow({'subject_id': str(i), 'seq': ' '.join(str(i) for i in k if i is not None)})
    csvFile.close()