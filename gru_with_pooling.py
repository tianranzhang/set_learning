
import numpy as np
import pandas as pd
import argparse
import gensim
import tensorflow
#tensorflow.config.experimental_run_functions_eagerly(False)
#tf.config.experimental_run_functions_eagerly(True)
import csv
import os
import uuid
from sklearn.metrics import roc_auc_score, average_precision_score,precision_score, recall_score,roc_curve,auc
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense,Input,Conv1D,MaxPooling1D,Dropout,LeakyReLU,LSTM,GRU,Embedding, Reshape, Dot, Multiply,Lambda,GlobalAveragePooling1D,Activation
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
#from keras.preprocessing import sequence
import keras_metrics
import tensorflow.keras.backend as K
#import tensorflow.keras.losses.BinaryCrossentropy

from tensorflow.keras.layers import Layer
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


def mean_loss(y_true, y_pred):
	return K.mean(y_pred, y_true)


class CustomRegularization(Layer):
    def __init__(self, **kwargs):
        super(CustomRegularization, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ld=x[0]
        rd=x[1]
        mae = tensorflow.keras.losses.MeanAbsoluteError(rd, ld)
        loss2 = K.sum(mae)
        self.add_loss(loss2,x)
        #you can output whatever you need, just update output_shape adequately
        #But this is probably useful
        return mae

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0],1)

class LossWithReg(tensorflow.keras.losses.Loss):

    def __init__(self, embs, reduction=tensorflow.keras.losses.Reduction.AUTO, name='LossWithReg'):
        super().__init__(reduction=reduction, name=name)
        self.embs = embs

    def call(self, y_true, y_pred):
        return tensorflow.keras.losses.binary_crossentropy(y_true, y_pred) + 0.01*tensorflow.keras.losses.MSE(self.embs[1], self.embs[0])#K.sum(self.embs[1] - self.embs[0], axis = -1)#K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)


# Define custom loss
def loss_with_l2(layer_pred,layer_true, wt = 0.01):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        return K.binary_crossentropy + wt*K.sqrt(K.sum(K.square(y_true - y_pred), axis = -1))#K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)
   
    # Return a function
    return loss

def loss_with_l1(layer):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred) + 0.01*K.sum(layer[1] - layer[0], axis = -1)#K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)
   
    # Return a function
    return loss

def pad_trun_sequences(seq, len_seq):
    new_seq = list()
    print('pad_trun_sequences length ', len(seq),'...')
    if len_seq==-1:
    	len_seq = max([len(element) for element in seq])
    for i in seq:
        #print(len(i))
        i = [j for j in i if j not in ['', 'nan']]
        length= len(i)
        if length > len_seq:
            new_seq = new_seq + [i[length - len_seq - 1 : -1]]
        if length <= len_seq:
            new_seq =  new_seq + [['00000'] * (len_seq-length) + i]
               
    return new_seq          
            

def encoder(seq, word_index):
    seq_encoded = list()
    for i in seq:
        seq_encoded = seq_encoded + [[word_index[j] for j in i]]     #if j!='nan'   
    return seq_encoded



#def model(base, layer_num, dim)

def training(train_data,  val_data, test_data, onehot_train, onehot_val, onehot_test, embedding_matrix, curr_cross_val = 0, 
             dim = 256, len_seq = 50, cnn_dim = 200, ksize=2,res_block = 1, epoch_num = 20, lr = 0.001):
	# fix random seed for reproducibility
    np.random.seed(100)
    from tcn import TCN
    batch_size, timesteps, input_dim = None, len_seq, dim
  
    #import datetime
    #curr_run_time= datetime.datetime.now()
    #if not os.path.exists("model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/"):
        #os.mkdir("model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/")
    #filepath = "model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/saved-model-{epoch:02d}-{val_loss:.4f}.hdf5"
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    sequence_input = Input(shape = (len_seq,dim, ), dtype = 'float32')
    #sequence_input  = Activation('relu')(sequence_input)
    #sequence_input_2 = Input(shape = (len_seq, ), dtype = 'int32')

    #emb_layer = Embedding(embedding_matrix.shape[0], dim,
                            #weights = [embedding_matrix],
                            #input_length = len_seq,
                            #trainable = False)
    #embedded_sequences = emb_layer(sequence_input)
    #embedded_sequences_2 = emb_layer(sequence_input_2)

    gru_layer = GRU(cnn_dim,dropout = 0.2,return_sequences=False)
    o = gru_layer(sequence_input)
    #o_2 = gru_layer(embedded_sequences_2)
    #d_layer = Dense(cnn_dim, activation='tanh')
    #h = d_layer(o)
    #h_1 = d_layer(o_1)
    #h_2 = d_layer(o_2)
    #s = Reshape((1, cnn_dim))(GlobalAveragePooling1D()(h))
    #s_1 = Reshape((1, cnn_dim))(GlobalAveragePooling1D()(h_1))
    #s_2 = Reshape((1, cnn_dim))(GlobalAveragePooling1D()(h_2))
    #a = Dot(axes=-1)([o, s])
    #a_1 = Dot(axes=-1)([o_1, s_1])
    #a_2 = Dot(axes=-1)([o_2, s_2])
    #a = Reshape((len_seq, 1))(Activation('softmax')(a))
    #a_1 = Reshape((len_seq, 1))(Activation('softmax')(a_1))
    #a_2 = Reshape((len_seq, 1))(Activation('softmax')(a_2))
    #x = Multiply()([o, a])
    #x_1 = Multiply()([o_1, a_1])
    #x_2 = Multiply()([o_2, a_2])
    #x = Lambda(lambda z:z*len_seq)(x)
    #x_1 = Lambda(lambda z:z*len_seq)(x_1)
    #x_2 = Lambda(lambda z:z*len_seq)(x_2)
    # AVG
    #x = GlobalAveragePooling1D()(x)
    #x_1 = GlobalAveragePooling1D()(o_1)                       
    #x_2 = GlobalAveragePooling1D()(o_2)

    #sub = K.abs(tensorflow.keras.layers.Subtract()([h_1,h_2]))
    #sub = Reshape((cnn_dim, len_seq))(sub)
    #mae = cnn_dim*len_seq*GlobalAveragePooling1D()(sub)
    #mae = Reshape([1])(mae)
    #mae = Dense(1, activation='sigmoid')(mae)
    #mae = tensorflow.keras.layers.Flatten()(mae)

    output_layer = Dense(1, activation = 'sigmoid')
    o = output_layer(o)


    #o = tensorflow.keras.layers.Concatenate()([o1,o2])
    m = Model(inputs = sequence_input, outputs = o)
    #n = [m.get_layer('emb1').output, m.get_layer('emb1').output]

    #m.add_loss(LossWithReg(n))


    
    m.compile(optimizer = optimizers.Adam(lr=lr), loss = 'binary_crossentropy', metrics= ['accuracy'])#,tensorflow.keras.metrics.AUC(), tensorflow.keras.metrics.Precision(), tensorflow.keras.metrics.Recall()]) #
    print(m.summary())
    #plot_model(m, to_file='multiple_outputs.png')

   


    #print(train_data)
    #tsbd = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callbacks_list = [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    #print(np.asarray(onehot_val))
    m.fit(train_data, onehot_train, validation_data = (val_data,  onehot_val), epochs = 100, batch_size = 128, verbose = 2, callbacks=callbacks_list) #callbacks=callbacks_list #validation_data=([np.array(val_data_1), np.array(val_data_2)],  [np.array(onehot_val_1), np.array(onehot_val_1)-np.array(onehot_val_1)]), 


    #best_model= 0
    #best_auc = 0

    #for counter in np.arange(100):
       # print("training epoch ", counter)
    #sampling:
        #number_of_rows = np.array(train_data).shape[0]
        #random_indices = np.random.choice(number_of_rows, size=5426, replace=False)
        #train_data_sample = np.array(train_data)[random_indices, :]
        #onehot_train_sample = np.array(onehot_train)[random_indices]


        #number_of_rows = np.array(val_data).shape[0]
        #random_indices = np.random.choice(number_of_rows, size=904, replace=False)
        #val_data_sample = np.array(val_data)[random_indices, :]
        #onehot_val_sample = np.array(onehot_val)[random_indices]

        #m.fit(np.array(train_data_sample), np.array(onehot_train_sample), validation_data = (np.array(val_data_sample),  np.array(onehot_val_sample)), epochs = 1, batch_size = 128, verbose = 2, callbacks=callbacks_list) #callbacks=callbacks_list #validation_data=([np.array(val_data_1), np.array(val_data_2)],  [np.array(onehot_val_1), np.array(onehot_val_1)-np.array(onehot_val_1)]), 
	    #m.fit([np.array(train_data_1), np.array(train_data_2)], [np.array(onehot_train_1), np.array(onehot_train_1)-np.array(onehot_train_1)], 
	    #epochs=epoch_num, batch_size=64, verbose = 2)
        #predicted_val_updated = m.predict(np.array(val_data))
        #fpr, tpr, _ = roc_curve(onehot_val,predicted_val_updated)
        #val_auc = auc(fpr, tpr)
        #del predicted_val_updated
        #print('AUC: ', val_auc)
        
        #if val_auc>best_auc:
            #best_auc = val_auc
            #best_model = m
            #predicted_val = best_model.predict(np.array(val_data))
            #predicted_test = best_model.predict(np.array(test_data))
            #no_improve = 0
        #if no_improve<10:
            #no_improve = no_improve+1
        #else:
            #break;




    #print('I got to reach here!')

    predicted_val = m.predict(np.array(val_data).astype('float64'))
    predicted_test = m.predict(np.array(test_data).astype('float64'))

    #scores_val = m.evaluate([np.array(val_data_1),np.array(val_data_2)],  [np.array(onehot_val_1), np.zeros([len(onehot_val_1),1])], verbose=0)
    #print(scores_val)
    #scores_val = [i[0] for i in scores_val]
    #scores_test= m.evaluate([np.array(test_data_1),np.array(test_data_2)],  [np.array(onehot_test_1), np.zeros([len(onehot_test_1),1])], verbose=0)
    #scores_test = [i[0] for i in scores_test]
    # Final evaluation of the model
    #scores_val = m.evaluate([np.array(test_data_1),np.array(test_data_2)],  [np.array(onehot_test_1), np.array(onehot_test_2)], verbose=0)[0]
    #scores_test = m.evaluate(np.array(test_data),  np.array(onehot_test), verbose=0)

    res = [predicted_val, predicted_test]#, scores_val, scores_test
    return res



#Start of main script ....
def main_pipeline(perm = 'noperm', perm_file = 'None', lr = 0.001, epoch_num = 10, cnn_dim = 256,  len_seq = 256, skip_gram = 1, dim = 50, win_size = 20, run_num = 0):

	#####################################
	# Load train/val/test subject lists #
	#####################################

	train_list = pd.read_csv('train_list_with_labels.csv', sep = '\t') #set_learning/
	val_list = pd.read_csv('val_list_with_labels.csv', sep = '\t')  #set_learning/
	test_list = pd.read_csv('test_list_with_labels.csv', sep = '\t') #set_learning/


	######################################################
	# Load data and labels under three scenarios #
	######################################################

	#1) when no permutation
	if perm == 'noperm':

		#pooled_sequences = pd.read_csv("tcn_abnormlabs_baseline/"+"pooling_"+'sum'+".csv", sep = ',')
		#train_data = pd.read_csv('train_lab_abnorm_sc1.csv', sep = ',')
		#val_data = pd.read_csv('val_lab_abnorm_sc1.csv', sep = ',')



		
		####temp modification for testing the robustness of alpha data on alpha test set
		test_perm_sequences = pd.read_csv("test_lab_abnorm_sc1.csv", sep = ',' )
		#train_data = test_perm_sequences[test_perm_sequences.subject_id.isin(train_list.subject_id)]
		#val_data = test_perm_sequences[test_perm_sequences.subject_id.isin(val_list.subject_id)]
		
		####temp modification for testing the robustness of noperm data on perm test set
		#perm_sequences = pd.read_csv(perm_file+".csv")

		#test_perm_sequences = pd.read_csv('test_lab_abnorm_sc1.csv', sep=',')
		#val_data = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
		test_data = test_perm_sequences[test_perm_sequences.subject_id.isin(test_list.subject_id)]
		####### end of temp modification

		train_y = list(train_data.HF)
		val_y = list(val_data.HF)
		test_y = list(test_data.HF)

	#2) when permutation
	elif perm == 'both':
		train_data_1 = pd.read_csv('train_lab_abnorm_sc1.csv', sep = ',')
		val_data_1 = pd.read_csv('val_lab_abnorm_sc1.csv', sep = ',')
		
		train_y_1 = list(train_data_1.HF)		
		val_y_1 = list(val_data_1.HF)



		perm_sequences = pd.read_csv(perm_file+".csv")
		
		train_data_2 = perm_sequences[perm_sequences.subject_id.isin(train_list.subject_id)]
		val_data_2 = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
		
		train_y_2 = list(train_data_2.HF)
		val_y_2 = list(val_data_2.HF)



		#combine the two parts
		train_data = pd.concat([train_data_1[['subject_id','seq']], train_data_2[['subject_id','seq']]])
		val_data = pd.concat([val_data_1[['subject_id','seq']], val_data_2[['subject_id','seq']]])
		train_y = train_y_1 + train_y_2
		val_y = val_y_1 + val_y_2
		
		test_perm_sequences = pd.read_csv('test_lab_abnorm_sc1.csv', sep = ',')

		#Load the perm sequences
		#val_data = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
		test_data = test_perm_sequences[test_perm_sequences.subject_id.isin(test_list.subject_id)]

		del perm_sequences
		
		#val_y = list(val_data.HF)
		test_y = list(test_data.HF)



	else: #perm data only
		print('reading pool data...')
		#print(perm_file)
		perm_sequences = pd.read_pickle("tcn_abnormlabs_baseline/"+"pooling_"+'max'+".pkl")
		
#pd.read_csv("tcn_abnormlabs_baseline/"+"pooling_"+'sum'+".csv")
		#test_perm_sequences = pd.read_csv(perm_file+".csv")

		train_data = perm_sequences[perm_sequences.subject_id.isin([str(i) for i in train_list.subject_id])]
		#print(train_data.head())
		
		print('train data ready...')
		#val_data = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
		val_data = perm_sequences[perm_sequences.subject_id.isin([str(i) for i in val_list.subject_id])]
		print('val data ready...')

		#generating matching controls based on the original sequences

		#train_data_pre = pd.read_csv('train_lab_abnorm_sc1.csv', sep = ',')
		#val_data_pre = pd.read_csv('val_lab_abnorm_sc1.csv', sep = ',')
		#test_data_pre = pd.read_csv('test_lab_abnorm_sc1.csv', sep = ',')

		
		#write header row
		#with open("tcn_abnormlabs_baseline/"+"matched_100_alpha.csv", 'a', newline='') as csvFile:
			#writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
			#writer.writerow({'subject_id': 'subject_id', 'seq': 'seq','HF': 'HF'})
		#csvFile.close()

		#for index, row in train_data.iterrows():
			#new_row = train_data_pre[train_data_pre.subject_id.isin([row.subject_id])]
			#with open("tcn_abnormlabs_baseline/"+"matched_100_alpha.csv", 'a', newline='') as csvFile:
				#writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
				#writer.writerow({'subject_id': str(list(new_row.subject_id)[0]),'seq': str(list(new_row.seq)[0]),'HF': int(list(new_row.HF)[0])})
			#csvFile.close()
		#for index, row in val_data.iterrows():
			#new_row = val_data_pre[val_data_pre.subject_id.isin([row.subject_id])]
			#with open("tcn_abnormlabs_baseline/"+"matched_100_alpha.csv", 'a', newline='') as csvFile:
				#writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
				#writer.writerow({'subject_id': str(list(new_row.subject_id)[0]),'seq': str(list(new_row.seq)[0]),'HF': int(list(new_row.HF)[0])})
			#csvFile.close()

		test_data = perm_sequences[perm_sequences.subject_id.isin([str(i) for i in test_list.subject_id])]
		print('test data ready...')
		#perm_sequences = pd.read_csv('test_lab_abnorm_sc1.csv', sep = ',')#"tcn_abnormlabs_baseline/permutation_percent_0_1_unique0_orderingalpha_label.csv")
		#test_data = perm_sequences[perm_sequences.subject_id.isin(test_list.subject_id)]
		#del perm_sequences
		#print('test data ready...')
		#for index, row in test_data.iterrows():
			#new_row = test_data_pre[test_data_pre.subject_id.isin([row.subject_id])]
			#with open("tcn_abnormlabs_baseline/"+"matched_100_alpha.csv", 'a', newline='') as csvFile:
				#writer = csv.DictWriter(csvFile, fieldnames=['subject_id','seq','HF'])
				#writer.writerow({'subject_id': str(list(new_row.subject_id)[0]),'seq': str(list(new_row.seq)[0]),'HF': int(list(new_row.HF)[0])})
			#csvFile.close()
		train_y = np.array(list(train_data.HF))
		val_y = np.array(list(val_data.HF))
		test_y = np.array(list(test_data.HF))

		print("input1 label converted..")


	########################################
	# Map train/val/test data into vectors #
	########################################
	

	if perm == 'noperm':
		print('not supposed to be here 1')

		val_x = pad_trun_sequences([i.split(' ') for i in val_data.seq][:val_data.count()[0]], len_seq )
		del val_data
		train_x = pad_trun_sequences([i.split(' ') for i in train_data.seq][:train_data.count()[0]], len_seq )
		del train_data


	elif perm == 'perm':
		print("I am here!!!")
		#val_x = pad_trun_sequences([i.split(' ') for i in val_data.seq][:val_data.count()[0]], len_seq )
		#val_x = pd.DataFrame(([' '.join(element) for element in val_x])).rename_axis(None)
		#val_x.columns  = ['seq']
		#val_x.to_csv(perm_file+'_val_x_'+str(len_seq)+'.csv',  index=False)
		#val_x = pd.read_csv(perm_file+'_val_x_'+str(len_seq)+'.csv', sep = '\t')['seq'].tolist()
		#val_x = [element.split(' ') for element in val_x]
		#print("perm val data loaded")
		
        #del val_data 
	
		#train_x = pad_trun_sequences([i.split(' ') for i in train_data.seq][:train_data.count()[0]], len_seq )
		#train_x = pd.DataFrame(([' '.join(element) for element in train_x])).rename_axis(None)
		#train_x.columns  = ['seq']
		#train_x.to_csv(perm_file+'_train_x_'+str(len_seq)+'.csv',  index=False)
		#train_x = pd.read_csv(perm_file+'_train_x_'+str(len_seq)+'.csv', sep = '\t')['seq'].tolist()
		#train_x = [element.split(' ') for element in train_x]
		#print("perm train data loaded")
		#del train_data

		#matched_data = pd.read_csv("tcn_abnormlabs_baseline/matched_100_alpha.csv",sep = ',')

		#matched_test = matched_data[matched_data.subject_id.isin(test_list.subject_id)]
		#matched_train = matched_data[matched_data.subject_id.isin(train_list.subject_id)]
		#matched_val = matched_data[matched_data.subject_id.isin(val_list.subject_id)]

		#matched_train_x = pad_trun_sequences([i.split(' ') for i in matched_train.seq][:matched_train.count()[0]], len_seq )
		#matched_train_x = pd.DataFrame(([' '.join(element) for element in matched_train_x])).rename_axis(None)
		#matched_train_x.columns  = ['seq']
		#matched_train_x.to_csv(perm_file+'_matched_alpha_train_x_'+str(len_seq)+'.csv',  index=False)
		#matched_train_x = pd.read_csv(perm_file+'_matched_original_train_x_'+str(len_seq)+'.csv', sep = '\t')['seq'].tolist()
		#matched_train_x = [element.split(' ') for element in matched_train_x]

		#print("matched train seq loaded")



		#matched_val_x = pad_trun_sequences([i.split(' ') for i in matched_val.seq][:matched_val.count()[0]], len_seq )
		#matched_val_x = pd.DataFrame(([' '.join(element) for element in matched_val_x])).rename_axis(None)
		#matched_val_x.columns  = ['seq']
		#matched_val_x.to_csv(perm_file+'_matched_alpha_val_x_'+str(len_seq)+'.csv',  index=False)
		#matched_val_x = pd.read_csv(perm_file+'_matched_original_val_x_'+str(len_seq)+'.csv', sep = '\t')['seq'].tolist()
		#matched_val_x = [element.split(' ') for element in matched_val_x]
		
		#print("matched val seq loaded")

		#matched_test_x = pad_trun_sequences([i.split(' ') for i in matched_test.seq][:matched_test.count()[0]], len_seq )
		#matched_test_x = pd.DataFrame(([' '.join(element) for element in matched_test_x])).rename_axis(None)
		#matched_test_x.columns  = ['seq']
		#matched_test_x.to_csv(perm_file+'_matched_alpha_test_x_'+str(len_seq)+'.csv',  index=False)
		#matched_test_x = pd.read_csv(perm_file+'_matched_original_test_x_'+str(len_seq)+'.csv', sep = '\t')['seq'].tolist()
		#matched_test_x = [element.split(' ') for element in matched_test_x]
		

		#print("matched data loaded")



	
	
	#print("converting test data", test_data.count())
	#test_x = pad_trun_sequences([i.split(' ') for i in test_data.seq][:test_data.count()[0]], len_seq )
	#del test_data
	#print('train/val/test sequences ready...')
	
	#perm_sequences = pd.read_csv(perm_file + ".csv")
	#pad_perm_sequences = pad_trun_sequences([i.split(' ') for i in perm_sequences.seq][:perm_sequences.count()[0]], len_seq ) 


	#train_data = pd.read_csv('train_lab_abnorm_sc1.csv', sep = ',')
	#val_data = pd.read_csv('val_lab_abnorm_sc1.csv', sep = ',')
	#test_data = pd.read_csv('test_lab_abnorm_sc1.csv', sep = ',')
	#pad_ori_sequences = pad_trun_sequences([i.split(' ') for i in train_data.seq][:train_data.count()[0]], len_seq )+pad_trun_sequences([i.split(' ') for i in val_data.seq][:val_data.count()[0]], len_seq )+pad_trun_sequences([i.split(' ') for i in test_data.seq][:test_data.count()[0]], -1 ) 

	#_model = gensim.models.Word2Vec(pad_ori_sequences, sg=1, window = 5, iter=5, size= 256, min_count=1, workers=20)#train_x+val_x+test_x
	#_model.save("word2vec.model")
	#print(' model saved...')
	#_model = gensim.models.Word2Vec.load("word2vec.model")


	_model = gensim.models.Word2Vec.load("word2vec.model")
	print('w2v model loaded')
	embeddings_index = {}
	print(len(_model.wv.vocab))
	embedding_matrix = np.zeros((len(_model.wv.vocab) + 1, dim))

	for i, word in enumerate(_model.wv.vocab):
		coefs = np.asarray(_model.wv[word], dtype = 'float32')
		embeddings_index[word] = coefs
		embedding_matrix[i] = coefs
	print('Found %s word vectors.' % len(embeddings_index))#embedding_matrix.shape[0])


	#generate word -> word index mapping
	word_index = {}
	for i, word in enumerate(_model.wv.vocab):
		word_index[word] = i

	#map code sequence to code index sequence	
	
	

	if perm_file == 'None':
		train_x = encoder(train_x, word_index)
		val_x = encoder(val_x, word_index)
	elif perm == 'perm':
		#train_x = encoder(train_x, word_index)
		#pd.DataFrame(np.array(train_x)).to_csv(perm_file+'_train_x_encoded'+str(len_seq)+'.csv', index=False)
		#print("perm train data encoded saved")
		train_x = np.array(train_data.seq)#np.array(pd.read_csv(perm_file + '_train_x_encoded' + str(len_seq) + '.csv', sep = ','))
		#sample = list(train_data.seq)[0]
		#sample = sample.replace(']','').replace('[','').replace(' [','').replace('] ','').replace('\n',',')
		#print(list(train_data.seq))
		#val_x = encoder(val_x, word_index)
		#pd.DataFrame(np.array(val_x)).to_csv(perm_file+'_val_x_encoded'+str(len_seq)+'.csv', index=False)
		#print("perm val data encoded saved")
		val_x = np.array(val_data.seq)




		#matched_train_x = encoder(matched_train_x, word_index)
		#pd.DataFrame(np.array(matched_train_x)).to_csv(perm_file+'_matched_alpha_train_x_encoded'+str(len_seq)+'.csv', index=False)
		#print("matched train data encoded saved")
		#matched_train_x = pd.read_csv(perm_file + '_matched_original_train_x_encoded'+str(len_seq) + '.csv', sep = ',')
		#print(matched_train_x.head())
		#matched_val_x = encoder(matched_val_x, word_index)
		#pd.DataFrame(np.array(matched_val_x)).to_csv(perm_file+'_matched_alpha_val_x_encoded'+str(len_seq)+'.csv', index=False)
		#print("matched val data encoded saved")
		#matched_val_x = np.array(pd.read_csv(perm_file + '_matched_original_val_x_encoded' + str(len_seq) + '.csv', sep = ','))
		#print(np.zeros(len(val_y)))

		#matched_test_x = encoder(matched_test_x, word_index)
		#pd.DataFrame(np.array(matched_test_x)).to_csv(perm_file+'_matched_alpha_test_x_encoded'+str(len_seq)+'.csv', index=False)
		#print("perm val data encoded saved")
		#matched_test_x = np.array(pd.read_csv(perm_file + '_matched_original_test_x_encoded' + str(len_seq) + '.csv', sep = ','))

	test_x = np.array(test_data.seq)


	#################################
	# Train predictive model for HF #
	#################################
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
		train_y = np.array(train_y,dtype = int)
		val_y = np.array(val_y,dtype = int)
		test_y = np.array(test_y,dtype = int)
		train_x = np.array([i.astype('float64') for i in list(train_x)])
		val_x = np.array([i.astype('float64') for i in list(val_x)])
		test_x = np.array([i.astype('float64') for i in list(test_x)])
		#print(test_x)
		

		predicted_val,predicted_test = training(train_x, val_x, test_x, train_y, val_y, test_y, embedding_matrix,
			dim = dim, len_seq = len_seq, cnn_dim = cnn_dim, epoch_num = epoch_num, lr = lr)  #,scores_val,scores_test 


    #TODO: pick the threshold that gives the best F1
		acc_val.append(0)#scores_val[1]
		acc_test.append(0)#scores_test[1]
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
			tensorflow.gfile.MkDir("tcn_abnormlabs_baseline/")
		if not os.path.exists("tcn_abnormlabs_baseline/"+ "/cv"):
			tensorflow.gfile.MkDir("tcn_abnormlabs_baseline/"+ "/cv")
		if not os.path.exists("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"):
			tensorflow.gfile.MkDir("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/")

		#with open("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"+"test_pred"+str(uuid.uuid4())+".csv", 'a', newline='') as csvFile:   
			#writer = csv.DictWriter(csvFile, fieldnames=['predicted_test','true_test'])
			#writer.writerow({'predicted_test':'predicted_test','true_test':'true_test'})
			#for i in range(len(test_y)):
				#writer.writerow({'predicted_test':predicted_test[i],'true_test':test_y[i]})
		#csvFile.close()

		#with open("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"+"validation_pred"+str(uuid.uuid4())+".csv", 'a', newline='') as csvFile:   
			#writer = csv.DictWriter(csvFile, fieldnames=['predicted_val','true_val'])
			#writer.writerow({'predicted_val':'predicted_val','true_val':'true_val'})
			#for i in range(len(val_y)):
				#writer.writerow({'predicted_val':predicted_val[i],'true_val':val_y[i]})
		#csvFile.close()
	
	with open("tcn_abnormlabs_baseline/"+"exp_logs_vanilla_gru_max_pooling.csv", 'a', newline='') as csvFile:
		writer = csv.DictWriter(csvFile, fieldnames=['acc_val','auc_vals','prec_val','rec_val', 'acc_test','auc_test','prec_test','rec_test',"prauc_vals","prauc_test","dim",
    		"cnn_dim","len_seq", "perm","lr","epoch_num","len_seq", "curr_dim","win_size", "perm_file","note","run_num"])
		writer.writerow({'acc_val': str(np.mean(acc_val)),'auc_vals': str(np.mean(auc_vals)),'prec_val': str(np.mean(prec_val)),'rec_val': str(np.mean(rec_val)),
                    	'acc_test': str(np.mean(acc_test)),'auc_test': str(np.mean(auc_test)),'prec_test': str(np.mean(prec_test)),'rec_test': str(np.mean(rec_test)), 
                    	"prauc_vals":str(np.mean(prauc_vals)),"prauc_test":str(np.mean(prauc_test)),
                    	'dim':str(dim),"cnn_dim":str(cnn_dim),"len_seq":str(len_seq), 'perm': perm, "lr": str(lr), "epoch_num":epoch_num, 
                    	"len_seq": str(len_seq), "curr_dim":str(dim),  "win_size": str(win_size), "perm_file": str(perm_file),"note":'GRU + max_pooling', "run_num": str(run_num)})
	csvFile.close()





import gc
run_num = 0
lr = 0.001
skip_gram = 1
epoch_num = 100

win_size = 5
dim = 256
#perm_file = 'None'
#res_block = 1
#for win_size in [5,10,20]:
for len_seq in [1095]:#128,256,
	for perm in ['perm']:#'perm', 
		for cnn_dim in [512, 256,128,64]:#,64,256   #{512,64} {256,64}
			for perm_file in ['tcn_abnormlabs_baseline/permutation_percent_1_100_unique0_label']:#'tcn_abnormlabs_baseline/permutation_percent_0_1_unique0_orderingalpha_label']:#,'tcn_abnormlabs_baseline/permutation_1_10_label','tcn_abnormlabs_baseline/permutation_1_6_label','tcn_abnormlabs_baseline/permutation_1_1_label', 'tcn_abnormlabs_baseline/permutation_1_2_label']:		
				for iterator in [1,2,3,4,5,6,7,8,9,10]:
					#for perm_file in ['tcn_abnormlabs_baseline/permutation_1_10_label','tcn_abnormlabs_baseline/permutation_1_6_label','tcn_abnormlabs_baseline/permutation_1_1_label', 'tcn_abnormlabs_baseline/permutation_1_2_label']:
					run_num = run_num+1
					print("iteration: ", iterator, "  run_num: ", run_num)
					if run_num < 27:
						continue;
					if perm=='noperm':
						#epoch_num = 100
						perm_file = 'None'
					#else: epoch_num = 5
					main_pipeline (perm = perm, perm_file = perm_file, lr = lr, epoch_num = epoch_num, cnn_dim = cnn_dim, 
						len_seq = len_seq, skip_gram = skip_gram, dim = dim, win_size = win_size, run_num = run_num)
					tensorflow.keras.backend.clear_session()
					gc.collect()



#o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid', dilation_rate=1,  activation=None, use_bias=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid',dilation_rate=1,  activation=None, use_bias=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid', dilation_rate=1, activation=None, use_bias=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = TCN(nb_filters = cnn_dim, kernel_size= ksize, dilations = [1,2,4,8],  nb_stacks = res_block, dropout_rate = 0.3, return_sequences=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = TCN(nb_filters = cnn_dim, kernel_size= ksize, dilations = [1,2,4],  nb_stacks = res_block, return_sequences=False)(embedded_sequences)
