import numpy as np
import pandas as pd
import argparse
import gensim
import tensorflow
#tf.config.experimental_run_functions_eagerly(True)
import csv
import os
import uuid
from sklearn.metrics import roc_auc_score, average_precision_score,precision_score, recall_score,roc_curve,auc
from tensorflow.keras.layers import Dense,Input,Conv1D,MaxPooling1D,Dropout,LeakyReLU,LSTM,GRU,Embedding, Reshape, Dot, Multiply,Lambda,GlobalAveragePooling1D,Activation
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
#from keras.preprocessing import sequence
import keras_metrics


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
            

def encoder(seq, word_index):
    seq_encoded = list()
    for i in seq:
        seq_encoded = seq_encoded + [[word_index[j] for j in i ]]     #if j!='nan'   
    return seq_encoded



#def model(base, layer_num, dim)

def training(train_data, val_data, test_data, onehot_train, onehot_val, onehot_test, embedding_matrix, curr_cross_val = 0, 
             dim = 256, len_seq = 50, cnn_dim = 200, ksize=2,res_block = 1, epoch_num = 20, lr = 0.001):
  # fix random seed for reproducibility
    np.random.seed(100)
    from tcn import TCN
    batch_size, timesteps, input_dim = None, len_seq, dim
  
  
    #ksize=2
    sequence_input = Input(shape=(len_seq,), dtype='int32')
    embedded_sequences = Embedding(embedding_matrix.shape[0], dim,
                            weights=[embedding_matrix],
                            input_length=len_seq,
                            trainable=False)(sequence_input)    
    #o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid', dilation_rate=1,  activation=None, use_bias=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid',dilation_rate=1,  activation=None, use_bias=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = Conv1D(cnn_dim, kernel_size=ksize, strides=ksize, padding='valid', dilation_rate=1, activation=None, use_bias=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = TCN(nb_filters = cnn_dim, kernel_size= ksize, dilations = [1,2,4,8],  nb_stacks = res_block, dropout_rate = 0.3, return_sequences=True)(embedded_sequences)
    #o = LeakyReLU(alpha=0.3)(o)
    #o = TCN(nb_filters = cnn_dim, kernel_size= ksize, dilations = [1,2,4],  nb_stacks = res_block, return_sequences=False)(embedded_sequences)
    o = GRU(cnn_dim,dropout = 0.2,return_sequences=True)(embedded_sequences)
    h = Dense(cnn_dim, activation='tanh')(o)
    s = Reshape((1, cnn_dim))(GlobalAveragePooling1D()(h))
    a = Dot(axes=-1)([o, s])
    a = Reshape((len_seq, 1))(Activation('softmax')(a))
    x = Multiply()([o, a])
    x = Lambda(lambda z:z*len_seq)(x)
    # AVG
    x = GlobalAveragePooling1D()(x)
    #o = tensorflow.keras.layers.Concatenate()([o1,o2])
    o = Dense(1, activation='sigmoid')(x)

    m = Model(inputs=[sequence_input], outputs=[o])
    m.compile(optimizer=optimizers.Adam(lr=lr),loss='binary_crossentropy', metrics= ['accuracy',tensorflow.keras.metrics.AUC(), tensorflow.keras.metrics.Precision(), tensorflow.keras.metrics.Recall()])
    print(m.summary())
  #import datetime
  #curr_run_time= datetime.datetime.now()
    #if not os.path.exists("model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/"):
        #os.mkdir("model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/")
    #filepath = "model_checkpoints/tcn/"+target+"/"+str(run_num)+ "/cv"+ str(curr_cross_val)+"/saved-model-{epoch:02d}-{val_loss:.4f}.hdf5"
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    
    
    #tsbd = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callbacks_list = [tensorflow.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience= 10, restore_best_weights = True)]
    #print(np.asarray(onehot_val))

    m.fit(np.array(train_data),  np.array(onehot_train), validation_data=(np.array(val_data),  np.array(onehot_val)), epochs=epoch_num, batch_size=128, verbose = 2, callbacks= callbacks_list)
  
    predicted_val = m.predict(np.array(val_data))
    predicted_test = m.predict(np.array(test_data))
    scores_val = m.evaluate(np.array(val_data),  np.array(onehot_val), verbose=0)
    scores_test = m.evaluate(np.array(test_data),  np.array(onehot_test), verbose=0)
    # Final evaluation of the model
    scores_val = m.evaluate(np.array(val_data),  np.array(onehot_val), verbose=0)
    scores_test = m.evaluate(np.array(test_data),  np.array(onehot_test), verbose=0)
    return [predicted_val, predicted_test,scores_val,scores_test]




#Start of main script ....
def main_pipeline (perm = 'noperm', perm_file = 'None', lr = 0.001, epoch_num = 10, cnn_dim = 256,  len_seq = 256, skip_gram = 1, dim = 50, win_size = 20, run_num = 0):

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

		
		train_data = pd.read_csv('train_lab_abnorm_sc1.csv', sep = ',')
		val_data = pd.read_csv('val_lab_abnorm_sc1.csv', sep = ',')
		
		####temp modification for testing the robustness of noperm data on perm test set
		
		#perm_sequences = pd.read_csv(perm_file+".csv")

		test_perm_sequences = pd.read_csv('tcn_abnormlabs_baseline/permutation_percent_1_1_label.csv', sep=',')
		#val_data = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
		test_data = test_perm_sequences[test_perm_sequences.subject_id.isin(test_list.subject_id)]
		del test_perm_sequences
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
		
		test_perm_sequences = pd.read_csv('tcn_abnormlabs_baseline/permutation_percent_1_1_label.csv', sep=',')

		#Load the perm sequences
		#val_data = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
		test_data = test_perm_sequences[test_perm_sequences.subject_id.isin(test_list.subject_id)]

		del perm_sequences
		del test_perm_sequences
		
		#val_y = list(val_data.HF)
		test_y = list(test_data.HF)



	else: #perm data only

		perm_sequences = pd.read_csv(perm_file+".csv")
		#test_perm_sequences = pd.read_csv(perm_file+".csv")

		train_data = perm_sequences[perm_sequences.subject_id.isin(train_list.subject_id)]
		#val_data = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
		val_data = perm_sequences[perm_sequences.subject_id.isin(val_list.subject_id)]
		
		test_perm_sequences = pd.read_csv('tcn_abnormlabs_baseline/permutation_percent_1_1_label.csv', sep=',')
		test_data = test_perm_sequences[test_perm_sequences.subject_id.isin(test_list.subject_id)]

		del perm_sequences
		del test_perm_sequences

		train_y = list(train_data.HF)
		val_y = list(val_data.HF)
		test_y = list(test_data.HF)


	########################################
	# Map train/val/test data into vectors #
	########################################

	train_x = pad_trun_sequences([i.split(' ') for i in train_data.seq][:train_data.count()[0]], len_seq )
	del train_data
	val_x = pad_trun_sequences([i.split(' ') for i in val_data.seq][:val_data.count()[0]], len_seq )
	del val_data 
	test_x = pad_trun_sequences([i.split(' ') for i in test_data.seq][:test_data.count()[0]], len_seq )
	del test_data
	
	#perm_sequences = pd.read_csv(perm_file + ".csv")
	#pad_perm_sequences = pad_trun_sequences([i.split(' ') for i in perm_sequences.seq][:perm_sequences.count()[0]], len_seq ) 


	#train_data = pd.read_csv('train_lab_abnorm_sc1.csv', sep = ',')
	#val_data = pd.read_csv('val_lab_abnorm_sc1.csv', sep = ',')
	#test_data = pd.read_csv('test_lab_abnorm_sc1.csv', sep = ',')
	
	#ori_sequences = pd.read_csv("all_train.csv")
	#pad_ori_sequences = pad_trun_sequences([i.split(' ') for i in train_data.seq][:train_data.count()[0]], len_seq )+pad_trun_sequences([i.split(' ') for i in val_data.seq][:val_data.count()[0]], len_seq )+pad_trun_sequences([i.split(' ') for i in test_data.seq][:test_data.count()[0]], len_seq ) 

	#del perm_sequences

	#if perm == 'noperm':
		#sequences = pad_ori_sequences

	#elif perm == 'both':	
		#sequences = pad_ori_sequences + pad_perm_sequences

	#else: #othe perm only cases...
		#sequences = pad_perm_sequences

	_model = gensim.models.Word2Vec(train_x+val_x+test_x, sg=1, window = win_size, iter=5, size= dim, min_count=1, workers=20)


	embeddings_index = {}
	embedding_matrix = np.zeros((len(_model.wv.vocab) + 1, dim))

	for i, word in enumerate(_model.wv.vocab):
		coefs = np.asarray(_model.wv[word], dtype='float32')
		embeddings_index[word] = coefs
		embedding_matrix[i] = coefs
	print('Found %s word vectors.' % len(embeddings_index))#embedding_matrix.shape[0])


	#generate word -> word index mapping
	word_index = {}
	for i, word in enumerate(_model.wv.vocab):
		word_index[word] = i

	#map code sequence to code index sequence
	train_x = encoder(train_x, word_index)
	val_x = encoder(val_x, word_index)
	test_x = encoder(test_x, word_index)

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
		predicted_val, predicted_test,scores_val,scores_test = training(train_x, val_x, test_x, train_y, val_y, test_y, embedding_matrix,
			dim = dim, len_seq = len_seq, cnn_dim = cnn_dim, epoch_num = epoch_num, lr = lr)
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
			tensorflow.gfile.MkDir("tcn_abnormlabs_baseline/")
		if not os.path.exists("tcn_abnormlabs_baseline/"+ "/cv"):
			tensorflow.gfile.MkDir("tcn_abnormlabs_baseline/"+ "/cv")
		if not os.path.exists("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/"):
			tensorflow.gfile.MkDir("tcn_abnormlabs_baseline/"+ "/cv"+ str(curr_cross_val)+"/")

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
	
	with open("tcn_abnormlabs_baseline/"+"exp_logs_gru_att_test_on_all_stamps_p1_1.csv", 'a', newline='') as csvFile: 
		writer = csv.DictWriter(csvFile, fieldnames=['acc_val','auc_vals','prec_val','rec_val', 'acc_test','auc_test','prec_test','rec_test',"prauc_vals","prauc_test","dim",
    		"cnn_dim","len_seq", "perm","lr","epoch_num","len_seq", "curr_dim","win_size", "perm_file","run_num"])
		writer.writerow({'acc_val': str(np.mean(acc_val)),'auc_vals': str(np.mean(auc_vals)),'prec_val': str(np.mean(prec_val)),'rec_val': str(np.mean(rec_val)),
                    	'acc_test': str(np.mean(acc_test)),'auc_test': str(np.mean(auc_test)),'prec_test': str(np.mean(prec_test)),'rec_test': str(np.mean(rec_test)), 
                    	"prauc_vals":str(np.mean(prauc_vals)),"prauc_test":str(np.mean(prauc_test)),
                    	'dim':str(dim),"cnn_dim":str(cnn_dim),"len_seq":str(len_seq), 'perm': perm, "lr": str(lr), "epoch_num":epoch_num, 
                    	"len_seq": str(len_seq), "curr_dim":str(dim),  "win_size": str(win_size), "perm_file": str(perm_file), "run_num": str(run_num)})
	csvFile.close()





import gc
run_num=0
lr = 0.001
skip_gram = 1
epoch_num = 30
#perm_file = 'None'
#res_block = 1
for cnn_dim in [128,64,256]:
	for len_seq in [256,128,512]:
		for dim in [64,32,128,256]:
			for win_size in [5,10,20]:
				for perm in ['noperm']:
					#for perm_file in ['tcn_abnormlabs_baseline/permutation_percent_1_1_label','tcn_abnormlabs_baseline/permutation_1_10_label','tcn_abnormlabs_baseline/permutation_1_6_label','tcn_abnormlabs_baseline/permutation_1_1_label', 'tcn_abnormlabs_baseline/permutation_1_2_label']:		
					#for perm_file in ['tcn_abnormlabs_baseline/permutation_1_10_label','tcn_abnormlabs_baseline/permutation_1_6_label','tcn_abnormlabs_baseline/permutation_1_1_label', 'tcn_abnormlabs_baseline/permutation_1_2_label']:
					run_num = run_num+1
					print("run_num ", run_num)
					main_pipeline (perm = perm, perm_file = 'None', lr = lr, epoch_num = epoch_num, cnn_dim = cnn_dim, 
						len_seq = len_seq, skip_gram = skip_gram, dim = dim, win_size = win_size, run_num = run_num)
					tensorflow.keras.backend.clear_session()
					gc.collect()