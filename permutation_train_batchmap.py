




import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import argparse
import gensim
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)




parser = argparse.ArgumentParser()
#add list of args
parser.add_argument('--perm', type=str, default= 'noperm')
parser.add_argument('--perm_file', type=str, default= 'None') #only useful when perm mode is 'both'

parser.add_argument('--lr', type=float, default= 0.001)
parser.add_argument('--epoch_num', type=int, default= 10)
parser.add_argument('--cnn_dim', type=int, default= 256)
parser.add_argument('--len_seq', type=int, default= 256)
parser.add_argument('--curr_dim', type=int, default= 50)
parser.add_argument('--win_size', type=int, default= 20)


args = parser.parse_args()

#Todo: add arguments and wrap this script into a function
#Todo: write a grid search script
perm = args.perm

#hard code settings for now
len_seq = args.len_seq
skip_gram = 1
win_size = args.win_size
dim = args.curr_dim


#Helper functions
def w2v_mapping(sequences, _model,len_seq):
    mapped = []
    for i in sequences:
        mapped.append([_model.wv[j].tolist() for index, j in enumerate(i) if index>len(i)-len_seq and j != 'nan'])
    mapped = np.array(pad_sequences(mapped, maxlen= len_seq, dtype='float', padding='pre', truncating='pre', value=0.0))
    return mapped

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    #value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true) #tf.contrib.metrics.streaming_auc
    m = tf.keras.metrics.AUC(num_thresholds=100)
    m.update_state(y_true,y_pred)
    # find all variables created for this metric
    #metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]   
    # They will be initialized for new session.
    #for v in metric_vars:
        #tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    #with tf.control_dependencies([update_op]):
        #value = tf.identity(value)
    #m = tf.keras.metrics.AUC(num_thresholds = 100)
    #m.update_state(y_true, y_pred)
    return m.result().numpy()
    
def precision(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.compat.v1.metrics.precision( y_true,y_pred,)

    # find all variables created for this metric
    metric_vars = [i for i in tf.compat.v1.local_variables() if 'precision' in i.name.split('/')[1]]  
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
      
def recall(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.compat.v1.metrics.recall(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.compat.v1.local_variables() if 'recall' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value




from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict_proba(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict_proba(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

#roc = RocCallback(training_data=(X_train, y_train),
                  #validation_data=(X_test, y_test))




#Start of main script ....

#Load train/val/test subject list with labels
train_list = pd.read_csv('train_list_with_labels.csv', sep = '\t') #set_learning/
val_list = pd.read_csv('val_list_with_labels.csv', sep = '\t')  #set_learning/
test_list = pd.read_csv('test_list_with_labels.csv', sep = '\t') #set_learning/


#####temp
#perm_sequences = pd.read_csv("tcn_abnormlabs_baseline/permutation_1_1"+".csv")


#######################################
# Train w2v model under two scenarios #
#######################################
#with open("all_sequences_abnorm.txt", "r") as text_file:  #set_learning/
        #sequences = [i.replace('\n','') for i in text_file]   
#sequences = [i.split(' ') for i in sequences]
ori_sequences = pd.read_csv("all_train.csv")

if perm == 'noperm':
	sequences = [i.split(' ') for i in list(ori_sequences.seq)]
else: #othe perm only cases...
    perm_sequences = pd.read_csv(perm + ".csv")
    sequences = [i.split(' ') for i in list(perm_sequences.seq)]

    #sequences = [i.split(' ') for i in list(perm_sequences.seq)]
#1) when no permutation
#if perm == 'noperm':
    #with open("all_sequences_abnorm.txt", "r") as text_file:  #set_learning/
        #sequences = [i.replace('\n','') for i in text_file]   
    #sequences = [i.split(' ') for i in sequences]
#2) when permutation
#elif perm == 'both':
    #with open("all_sequences_abnorm.txt", "r") as text_file: #set_learning/
        #sequences = [i.replace('\n','') for i in text_file]
    #sequences_1 = [i.split(' ') for i in sequences]

    #perm_sequences = pd.read_csv(args.perm_file+".csv")
    #sequences_2 = [i.split(' ') for i in list(perm_sequences.seq)]

    #sequences = sequences_1 + sequences_2

#else: #othe perm only cases...
    #perm_sequences = pd.read_csv(perm+".csv")
    #sequences = [i.split(' ') for i in list(perm_sequences.seq)]

#_model = gensim.models.Word2Vec(sequences, sg=1, window = win_size, iter=5, size= dim, min_count=1, workers=20)
del sequences

######################################################
# Load train/val data and labels under two scenarios #
######################################################

#1) when no permutation
if perm == 'noperm':
	train_data = pd.read_csv('train_lab_abnorm_sc1.csv', sep = ',') #set_learning/
	#train_data = ori_sequences[ori_sequences.subject_id.isin(train_list.subject_id)] 
	val_data = pd.read_csv('val_lab_abnorm_sc1.csv', sep = ',') 

	#val_data = pd.read_csv('val_lab_abnorm.csv', sep = ',') #set_learning/
	test_data = pd.read_csv('test_lab_abnorm_sc1.csv', sep = ',')

	#train_y = list(train_list.HF)
	#val_y = list(val_list.HF)
	train_y = list(train_data.HF)#[int(train_list[train_list['subject_id']==i].HF) for i in train_data.subject_id]
	val_y = list(val_data.HF)#[int(val_list[val_list['subject_id']==i].HF) for i in val_data.subject_id]
	test_y = list(test_data.HF)#[int(test_list[test_list['subject_id']==i].HF) for i in test_data.subject_id]
    

#2) when permutation
elif perm == 'both':
    
	train_data_1 = pd.read_csv('train_lab_abnorm_sc1.csv', sep = ',')  #set_learning/
	val_data_1 = pd.read_csv('val_lab_abnorm_sc1.csv', sep = ',')#set_learning/
	
	train_y_1 = list(train_list.HF)
	val_y_1 = list(val_list.HF)

	train_data_2 = perm_sequences[perm_sequences.subject_id.isin(train_list.subject_id)]
	val_data_2 = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
	del perm_sequences

	train_y_2 = [int(train_list[train_list['subject_id']==i].HF) for i in train_data_2.subject_id]
	val_y_2 = [int(val_list[val_list['subject_id']==i].HF) for i in val_data_2.subject_id]
	#combine the two parts
	
	train_data = pd.concat([train_data_1[['subject_id','seq']], train_data_2[['subject_id','seq']]])
	val_data = pd.concat([val_data_1[['subject_id','seq']], val_data_2[['subject_id','seq']]])

	train_y = train_y_1 + train_y_2
	val_y = val_y_1 + val_y_2

else: #other perm cases...

	perm_sequences = pd.read_csv(perm+".csv")
	train_data = perm_sequences[perm_sequences.subject_id.isin(train_list.subject_id)]
	#train_data = pd.read_csv('train_lab_abnorm.csv', sep = ',')
	val_data = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
	test_data = ori_sequences[ori_sequences.subject_id.isin(test_list.subject_id)]

	del perm_sequences

	train_y = list(train_data.HF)#[int(train_list[train_list['subject_id']==i].HF) for i in train_data.subject_id]
	val_y = list(val_data.HF)#[int(val_list[val_list['subject_id']==i].HF) for i in val_data.subject_id]
	test_y = list(test_data.HF)#[int(test_list[test_list['subject_id']==i].HF) for i in test_data.subject_id]



#############################
# Load test data and labels #
#############################


#test_data = perm_sequences[perm_sequences.subject_id.isin(test_list.subject_id)]
#test_y = [int(test_list[test_list['subject_id']==i].HF) for i in test_data.subject_id]


#to make test set comparable to non-perm baseline
#test_data = pd.read_csv('test_lab_abnorm.csv', sep = ',')
#test_y = list(test_data.HF)


########################################
# Map train/val/test data into vectors #
########################################


def pad_trun_sequences(seq, len_seq):
    new_seq = list()
    for i in seq:
        #print(len(i))
        length= len(i)
        if length>len_seq:
             new_seq =  new_seq + [i[length-len_seq-1:-1]]
        if length<=len_seq:
             new_seq =  new_seq + [['00000']*(len_seq-length) +i]
                
    return new_seq          
            



train_x = pad_trun_sequences([i.split(' ') for i in train_data.seq][:train_data.count()[0]], len_seq )
#del train_data
val_x = pad_trun_sequences([i.split(' ') for i in val_data.seq][:val_data.count()[0]], len_seq )
#del val_data 
test_x = pad_trun_sequences([i.split(' ') for i in test_data.seq][:test_data.count()[0]], len_seq )
#del test_data
#[i.split(' ') for i in train_data.seq][0]

_model = gensim.models.Word2Vec(train_x+val_x+test_x, sg=1, window = win_size, iter=5, size= dim, min_count=1, workers=20)


embeddings_index = {}
embedding_matrix = np.zeros((len(_model.wv.vocab) + 1, dim))

for i, word in enumerate(_model.wv.vocab):
    coefs = np.asarray(_model.wv[word], dtype='float32')
    embeddings_index[word] = coefs
    embedding_matrix[i] = coefs
print('Found %s word vectors.' % len(embeddings_index))



word_index = {}
for i, word in enumerate(_model.wv.vocab):
    word_index[word] = i


def encoder(seq):
    seq_encoded = list()
    for i in seq:
        seq_encoded = seq_encoded + [[word_index[j] for j in i]]       
    return seq_encoded


train_x = encoder(train_x)
val_x = encoder(val_x)
test_x = encoder(test_x)




#generate mapped 
#train_x = w2v_mapping([i.split(' ') for i in train_data.seq], _model,len_seq)
del train_data
#val_x = w2v_mapping([i.split(' ') for i in val_data.seq], _model,len_seq)
del val_data 
#test_x = w2v_mapping([i.split(' ') for i in test_data.seq], _model,len_seq)
del test_data



#################################
# Train predictive model for HF #
#################################

#RNN model
#import tensorflow as tf
#from tf import keras
#from keras import optimizers
#from tf.keras import optimizers
#from keras.models import Sequential
#from keras.layers import Dense,Input
#from keras.layers import LSTM,GRU
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence
#import keras_metrics
#from keras.callbacks import ModelCheckpoint, CSVLogger
#from keras.callbacks import TensorBoard
#from tcn import TCN


def training(train_data, val_data, test_data, onehot_train, onehot_val, onehot_test, curr_cross_val = 0, 
             dim = 256, len_seq = 50, cnn_dim = 200, epoch_num = 20, lr = 0.001):
  # fix random seed for reproducibility
    np.random.seed(100)
  #model = Sequential()
  #if (lstm_num == 1):
    #model.add(LSTM(lstm_dim, input_shape=(200,dim)))
  #else:
    #model.add(LSTM(lstm_dim, input_shape=(200,dim), return_sequences=True))
    #model.add(LSTM(int(lstm_dim/2)))
  #model.add(Dense(target_num, activation='sigmoid'))
  #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',auc_roc, precision, recall])#keras_metrics.precision(),keras_metrics.recall()])
  #print(model.summary())
    #import keras
    #import tensorflow as tf

    from tensorflow.keras.layers import Dense,Input,Conv1D,MaxPooling1D,Dropout,LeakyReLU,LSTM,GRU,Embedding
    #from keras.models import Input, Model
    from tensorflow.keras.models import Model

    
    #from tf import keras
    #from keras import optimizers
    from tensorflow.keras import optimizers
    #from keras.models import Sequential
    #from keras.layers import Dense,Input
    #from keras.layers import LSTM,GRU
    #from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    import keras_metrics
    #from keras.callbacks import ModelCheckpoint, CSVLogger
    #from keras.callbacks import TensorBoard
    from tcn import TCN

    #from tcn import TCN
    batch_size, timesteps, input_dim = None, len_seq, dim
  
  
    ksize=2
    sequence_input = Input(shape=(len_seq,), dtype='int32')
    embedded_sequences = Embedding(len(embeddings_index) + 1, dim,
                            weights=[embedding_matrix],
                            input_length=len_seq,
                            trainable=False)(sequence_input)    
    #o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid', dilation_rate=1,  activation=None, use_bias=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid',dilation_rate=1,  activation=None, use_bias=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid', dilation_rate=1, activation=None, use_bias=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = TCN(dropout_rate=0.3, return_sequences=False)(embedded_sequences)

    #o = TCN(nb_filters=cnn_dim, return_sequences=True)(embedded_sequences)
    o = GRU(cnn_dim,dropout = 0.2)(embedded_sequences)
    o = Dense(1,activation='sigmoid')(o)

    m = Model(inputs=[sequence_input], outputs=[o])


    #metrics = [RocCallback()]
    m.compile(optimizer=optimizers.Adam(lr=lr),loss='binary_crossentropy', metrics= ['accuracy',tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
  
  #import datetime
  #curr_run_time= datetime.datetime.now()
    #if not os.path.exists("model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/"):
        #os.mkdir("model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/")
    #filepath = "model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/saved-model-{epoch:02d}-{val_loss:.4f}.hdf5"
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    
    
    #tsbd = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    #callbacks_list = [RocCallback(training_data=(train_data, onehot_train),
                  #validation_data=(val_data, onehot_val))]
  
    m.fit(np.array(train_data),  np.array(onehot_train), validation_data=(np.array(val_data),  np.array(onehot_val)), epochs=epoch_num, batch_size=128, verbose = 2)#,callbacks= callbacks_list)

    #print(m.history)

    #h = model.fit(train_data, onehot_train,validation_data=(val_data, onehot_val), epochs=epoch_num, batch_size=64,callbacks=callbacks_list)
  
    predicted_val = m.predict(np.array(val_data))
    predicted_test = m.predict(np.array(test_data))
    scores_val = m.evaluate(np.array(val_data),  np.array(onehot_val), verbose=0)
    scores_test = m.evaluate(np.array(test_data),  np.array(onehot_test), verbose=0)
    # Final evaluation of the model
    scores_val = m.evaluate(np.array(val_data),  np.array(onehot_val), verbose=0)
    scores_test = m.evaluate(np.array(test_data),  np.array(onehot_test), verbose=0)
    return [predicted_val, predicted_test,scores_val,scores_test]




import csv
import os
import uuid
curr_dim = args.curr_dim
len_seq = args.len_seq
cnn_dim = args.cnn_dim
epoch_num = args.epoch_num
acc_val = []
acc_test = []
auc_vals = []
auc_test = []
prauc_vals = []
prauc_test = []
prec_val = []
prec_test = []
rec_val = []
rec_test = []
for curr_cross_val in np.arange(1):
    #train
    #making directories to save models/predictions for val/test
    predicted_val, predicted_test,scores_val,scores_test = training(train_x, val_x, test_x, train_y, val_y, test_y, 
                                                curr_cross_val, args.curr_dim,args.len_seq, args.cnn_dim, args.epoch_num, args.lr)
    from sklearn.metrics import roc_auc_score, average_precision_score,precision_score, recall_score,roc_curve,auc
    #TODO: pick the threshold that gives the best F1
    acc_val.append(scores_val[1])
    acc_test.append(scores_test[1])
    fpr, tpr, _ = roc_curve(val_y,predicted_val)
    auc_vals.append(auc(fpr, tpr))
    fpr, tpr, _ = roc_curve(test_y,predicted_test)
    auc_test.append(auc(fpr, tpr))
    prec_val.append(precision_score(val_y,(predicted_val>0.5)*1))
    prec_test.append(precision_score(test_y,(predicted_test>0.5)*1))
    rec_val.append(recall_score(val_y,(predicted_val>0.5)*1))
    rec_test.append(recall_score(test_y,(predicted_test>0.5)*1))
    prauc_vals.append(average_precision_score(val_y,predicted_val))
    prauc_test.append(average_precision_score(test_y,predicted_test))
    if not os.path.exists("tcn_abnormlabs_baseline/"):
        tf.gfile.MkDir("tcn_abnormlabs_baseline/")
    if not os.path.exists("tcn_abnormlabs_baseline/"+ "/cv"):
        tf.gfile.MkDir("tcn_abnormlabs_baseline/"+ "/cv")
    if not os.path.exists("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"):
        tf.gfile.MkDir("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/")
    #np.savetxt("set_learning/tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"+"validation_pred"+str(uuid.uuid4())+".csv", predicted_val, delimiter=",")
    #np.savetxt("set_learning/tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"+"test_pred"+str(uuid.uuid4())+".csv", predicted_test, delimiter=",")

    with open("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"+"test_pred"+str(uuid.uuid4())+".csv", 'a', newline='') as csvFile:   
        writer = csv.DictWriter(csvFile, fieldnames=['predicted_test','true_test'])
        writer.writerow({'predicted_test':'predicted_test','true_test':'true_test'})
        for i in range(len(test_y)):
            writer.writerow({'predicted_test':predicted_test[i],'true_test':test_y[i]})
    csvFile.close()

    with open("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"+"validation_pred"+str(uuid.uuid4())+".csv", 'a', newline='') as csvFile:   
        writer = csv.DictWriter(csvFile, fieldnames=['predicted_val','true_val'])
        writer.writerow({'predicted_val':'predicted_val','true_val':'true_val'})
        for i in range(len(val_y)):
            writer.writerow({'predicted_val':predicted_val[i],'true_val':val_y[i]})
    csvFile.close()
	
with open("tcn_abnormlabs_baseline/"+"exp_logs.csv", 'a', newline='') as csvFile:   #set_learning/
    writer = csv.DictWriter(csvFile, fieldnames=['acc_val','auc_vals','prec_val','rec_val', 'acc_test','auc_test','prec_test','rec_test',"prauc_vals","prauc_test","dim",
    	"cnn_dim","len_seq", "perm","lr","epoch_num","len_seq", "curr_dim","win_size", "perm_file"])
    writer.writerow({'acc_val': str(np.mean(acc_val)),'auc_vals': str(np.mean(auc_vals)),'prec_val': str(np.mean(prec_val)),'rec_val': str(np.mean(rec_val)),
                    'acc_test': str(np.mean(acc_test)),'auc_test': str(np.mean(auc_test)),'prec_test': str(np.mean(prec_test)),'rec_test': str(np.mean(rec_test)), 
                    "prauc_vals":str(np.mean(prauc_vals)),"prauc_test":str(np.mean(prauc_test)),
                    'dim':str(dim),"cnn_dim":str(cnn_dim),"len_seq":str(len_seq), 'perm': perm, "lr": str(args.lr), "epoch_num":args.epoch_num, 
                    "len_seq": str(args.len_seq), "curr_dim":str(args.curr_dim),"win_size": str(args.win_size), "perm_file": str(args.perm_file)})
csvFile.close()
