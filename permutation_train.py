import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
def w2v_mapping(sequences, _model,len_seq):
    mapped = []
    for i in sequences:
        mapped.append([_model.wv[j].tolist() for index, j in enumerate(i) if index>len(i)-len_seq])
    mapped = np.array(pad_sequences(mapped, maxlen= len_seq, dtype='float', padding='pre', truncating='pre', value=0.0))
    return mapped

perm = "permutation_1_10"
train_list = pd.read_csv('set_learning/train_list_with_labels.csv', sep = '\t')
val_list = pd.read_csv('set_learning/val_list_with_labels.csv', sep = '\t')
test_list = pd.read_csv('set_learning/test_list_with_labels.csv', sep = '\t')

perm_sequences = pd.read_csv(perm+".csv")
perm_sequences.head()


import gensim
len_seq = 200
skip_gram = 1
win_size = 20
dim = 50

sequences = [i.split(' ') for i in list(perm_sequences.seq)]#[i.split(' ') for i in list(all_train.seq)]
_model = gensim.models.Word2Vec(sequences, sg=skip_gram, window = win_size, iter=5, size= dim, min_count=1, workers=20)
del sequences



#spliting train/val/test sets
train_data = perm_sequences[perm_sequences.subject_id.isin(train_list.subject_id)]
val_data = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
test_data = perm_sequences[perm_sequences.subject_id.isin(test_list.subject_id)]

#spliting train/val/test labels
train_y = [train_list[train_list['subject_id']==i].HF for i in list(perm_sequences.subject_id)]
val_y = [val_list[val_list['subject_id']==i].HF for i in list(perm_sequences.subject_id)]
test_y = [test_list[test_list['subject_id']==i].HF for i in list(perm_sequences.subject_id)]

del perm_sequences

train_x = w2v_mapping([i.split(' ') for i in train_data.seq], _model,len_seq)
del train_data
val_x = w2v_mapping([i.split(' ') for i in val_data.seq], _model,len_seq)
del val_data 
test_x = w2v_mapping([i.split(' ') for i in test_data.seq], _model,len_seq)
del test_data


#Train predictive model for HF
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]   
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
    
def precision(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.precision(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'precision' in i.name.split('/')[1]]  
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
      
def recall(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.recall(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'recall' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value



#RNN model
import numpy
from keras import optimizers
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.layers import LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras_metrics
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import TensorBoard


def training(train_data, val_data, test_data, onehot_train, onehot_val, onehot_test, curr_cross_val = 0, 
             dim = 256, len_seq = 50, cnn_dim = 200, epoch_num = 20):
  # fix random seed for reproducibility
    numpy.random.seed(100)
  #model = Sequential()
  #if (lstm_num == 1):
    #model.add(LSTM(lstm_dim, input_shape=(200,dim)))
  #else:
    #model.add(LSTM(lstm_dim, input_shape=(200,dim), return_sequences=True))
    #model.add(LSTM(int(lstm_dim/2)))
  #model.add(Dense(target_num, activation='sigmoid'))
  #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',auc_roc, precision, recall])#keras_metrics.precision(),keras_metrics.recall()])
  #print(model.summary())
    import keras
    from keras.layers import Dense,Conv1D,MaxPooling1D,Dropout
    from keras.models import Input, Model
  #from tcn import TCN
    batch_size, timesteps, input_dim = None, len_seq, dim
    from keras.layers import LeakyReLU
  
  
    ksize=3
    i = Input(batch_shape=(batch_size, timesteps, input_dim))
    o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid', dilation_rate=1,  activation=None, use_bias=True)(i)
    o = LeakyReLU(alpha=0.3)(o)
    o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid',dilation_rate=1,  activation=None, use_bias=True)(i)
    o = LeakyReLU(alpha=0.3)(o)
    o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid', dilation_rate=1, activation=None, use_bias=True)(i)
    o = LeakyReLU(alpha=0.3)(o)
    o = GRU(cnn_dim,dropout = 0.2)(o)
    o = Dense(1,activation='sigmoid')(o)

    m = Model(inputs=[i], outputs=[o])
    m.compile(optimizer=optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy',auc_roc, precision, recall])
  
  #import datetime
  #curr_run_time= datetime.datetime.now()
    #if not os.path.exists("model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/"):
        #os.mkdir("model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/")
    #filepath = "model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/saved-model-{epoch:02d}-{val_loss:.4f}.hdf5"
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    
    
    tsbd = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callbacks_list = [tsbd]
  
    m.fit(train_data, onehot_train, validation_data=(val_data, onehot_val), epochs=epoch_num, batch_size=128,callbacks=callbacks_list)

    #h = model.fit(train_data, onehot_train,validation_data=(val_data, onehot_val), epochs=epoch_num, batch_size=64,callbacks=callbacks_list)
  
    predicted_val = m.predict(val_data)
    predicted_test = m.predict(test_data)
    scores_val = m.evaluate(val_data, onehot_val, verbose=0)
    scores_test = m.evaluate(test_data, onehot_test, verbose=0)
    # Final evaluation of the model
    scores_val = m.evaluate(val_data, onehot_val, verbose=0)
    scores_test = m.evaluate(test_data, onehot_test, verbose=0)
    return [predicted_val, predicted_test,scores_val,scores_test]




import csv
import os
import uuid
curr_dim = 50
len_seq = 200
cnn_dim = 200
epoch_num = 10
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
    import tensorflow as tf       
    predicted_val, predicted_test,scores_val,scores_test = training(train_x, val_x, test_x, train_y, val_y, test_y, 
                                                curr_cross_val, curr_dim,len_seq, cnn_dim, epoch_num)
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
    if not os.path.exists("set_learning/tcn_abnormlabs_baseline/"):
        tf.gfile.MkDir("set_learning/tcn_abnormlabs_baseline/")
    if not os.path.exists("set_learning/tcn_abnormlabs_baseline/"+ "/cv"):
        tf.gfile.MkDir("set_learning/tcn_abnormlabs_baseline/"+ "/cv")
    if not os.path.exists("set_learning/tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"):
        tf.gfile.MkDir("set_learning/tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/")
    np.savetxt("set_learning/tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"+"validation_pred"+str(uuid.uuid4())+".csv", predicted_val, delimiter=",")
    np.savetxt("set_learning/tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"+"test_pred"+str(uuid.uuid4())+".csv", predicted_test, delimiter=",")
with open("set_learning/tcn_abnormlabs_baseline/"+"exp_logs.csv", 'a', newline='') as csvFile:   
    writer = csv.DictWriter(csvFile, fieldnames=['acc_val','auc_vals','prec_val','rec_val', 'acc_test','auc_test','prec_test','rec_test',"prauc_vals","prauc_test","dim","cnn_dim","len_seq", "perm"])
    writer.writerow({'acc_val': str(np.mean(acc_val)),'auc_vals': str(np.mean(auc_vals)),'prec_val': str(np.mean(prec_val)),'rec_val': str(np.mean(rec_val)),
                    'acc_test': str(np.mean(acc_test)),'auc_test': str(np.mean(auc_test)),'prec_test': str(np.mean(prec_test)),'rec_test': str(np.mean(rec_test)), 
                    "prauc_vals":str(np.mean(prauc_vals)),"prauc_test":str(np.mean(prauc_test)),
                    'dim':str(dim),"cnn_dim":str(cnn_dim),"len_seq":str(len_seq), 'perm': perm })
csvFile.close()
