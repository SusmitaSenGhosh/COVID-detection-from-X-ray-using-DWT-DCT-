# import packages
from numpy import loadtxt
import numpy as np
import random
from my_metrics import average_class_specific_accuracy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
import cv2
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import json
from tensorflow.python.keras import backend as K
import os
from plot_confusion_mat import *
from sklearn.utils import class_weight
from my_resnet import *
import json 
import matplotlib.pyplot as plt
# %% set paths and parameters
for k in range(0,5):
    txt_file_path = 'D:/Simpi/Work/COVID19/For_paper/txtfiles/'
    preprocessed_image_path = 'C:/Users/susmi/for_paper/preprocessed_merged/all_images/'
    weight_path ='C:/Users/susmi/for_paper/output/pixel/fold'+str(k+1)+'/'
    intermedite_output_path = 'C:/Users/susmi/for_paper/pixel_intermediate_fold' +str(k+1)+'/'
    batch_size = 32 
    num_classes = 3
    epochs = 300
    input_size = 224
    no_fold = 5
    
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if not os.path.exists(intermedite_output_path):
        os.makedirs(intermedite_output_path)
    #%%
    def reset_random_seeds(seed):
       os.environ['PYTHONHASHSEED']=str(seed)
       os.environ['TF_DETERMINISTIC_OPS'] = '1'
       tf.random.set_seed(seed)
       np.random.seed(seed)
       random.seed(seed)
    
    def get_intermediate_output(data_dir,text_file,intermediate_layer_model,savePath): 
        line_content= text_file
        for i in range(0,len(line_content)):
            file_name = line_content[i].split(" ")[1]
            x = np.asarray(cv2.imread(data_dir + file_name),dtype = np.float32)/255
            intermediate_output = intermediate_layer_model.predict(np.expand_dims(x, axis = 0))
            print(file_name)
            np.save(savePath+'/'+file_name,intermediate_output)
            
    def rotate(l, n):
        return l[n:] + l[:n]
    
    def get_data_label(data_dir,text_file): 
        x = []
        y = []
        weights = []
        mapping={
            'normal': 0,
            'pneumonia': 1,
            'COVID-19': 2
        }
        line_content= text_file
        for i in range(0,len(line_content)):
            file_name = line_content[i].split(" ")[1]
            x.append(cv2.imread(data_dir + file_name))
            y.append(line_content[i].split(" ")[2])
        y =list(map(mapping.get, y))
        for j in range(0,len(list(set(y)))):
            weights.append(y.count(list(set(y))[j])) 
        weights = [element / sum(weights) for element in weights]
        return np.asarray(x,dtype = np.float32)/255, to_categorical(np.asarray(y,dtype = np.float32)), weights
    
    
    #%% define model
    seed = 0#np.random.randint(0,1000)
    
    reset_random_seeds(seed)
    
    model = get_resnet(seed)
    
    #%% load data and train and save
    fold_set = list(range(0,5))
    fold_set = rotate(fold_set, k)
    train_files = []
    test_fold = fold_set[0]
    val_fold = fold_set[1]
    train_fold = fold_set[2::]
    with open(txt_file_path+'fold_'+str(test_fold+1)+'.txt', 'r') as fr:
         test_files = fr.readlines()
    with open(txt_file_path+'fold_'+str(val_fold+1)+'.txt', 'r') as fr:
         val_files = fr.readlines()
    for i in train_fold:
         with open(txt_file_path+'fold_'+str(i+1)+'.txt', 'r') as fr:
             train_files.extend(fr.readlines())
    
    
    train_generator = BalanceCovidDataset(data_dir=preprocessed_image_path,
                                    csv_file=train_files,
                                    batch_size=batch_size,
                                    input_shape=(input_size,input_size))
    
    val_generator = BalanceCovidDataset(data_dir=preprocessed_image_path,
                                    csv_file=val_files,
                                    batch_size=batch_size,
                                    input_shape=(input_size,input_size))
    x_test, y_test, weights_test = get_data_label(preprocessed_image_path,test_files)
    _, y_train,_ = get_data_label(preprocessed_image_path,train_files)
    _, y_val,_ = get_data_label(preprocessed_image_path,val_files)
    labelTr = np.argmax(np.concatenate((y_train,y_val)),axis = 1) 
    weights = class_weight.compute_class_weight('balanced', np.unique(labelTr),labelTr)
    classes = [0,1,2]
    class_weights = {classes[i]: weights[i] for i in range(len(classes))}
    del y_train, y_val
    #%% train
    # checkpoint = ModelCheckpoint(weight_path+'/checkpoint.h5', monitor='val_acsa', save_best_only=True, mode='max', verbose = 1)
    # earlystop = EarlyStopping(monitor='val_acsa', min_delta=0, patience=50, verbose=0, mode='max', baseline=None, restore_best_weights=False)
    # callbacks_list = [checkpoint,earlystop]
    # h = model.fit(x = train_generator, batch_size=batch_size, epochs=epochs, class_weight = class_weights, verbose=2,validation_data= val_generator, shuffle=True,callbacks = callbacks_list)
    # model_json = model.to_json()
    # with open(weight_path+'/model.json', "w") as json_file:
    #     json_file.write(model_json)
    # with open(weight_path+'/model_history.json', 'w') as f:
    #     json.dump(h.history, f)
    
    #%% evaluate
    
    model.load_weights(weight_path+'/checkpoint.h5')
    y_pred = model.predict(x = x_test)
    matrix = confusion_matrix(np.argmax(y_test, axis = 1),np.argmax(y_pred,axis = 1))
    matrix = matrix.astype('float')
    class_names = ['Normal', 'Pneumonia', 'COVID-19']
    title = "Confusion Matrix for fold#" +str(k+1)
    fig = plot_confusion_mat(cm=matrix, normalize = False, target_names = class_names, title = title)
    #fig.savefig(weight_path+'confusion_matrix', bbox_inches='tight')
    cm = matrix
    class_acc = [100*cm[i,i]/np.sum(cm[i,:]) if np.sum(cm[i,:]) else 0 for i in range(len(cm))]
    ppvs = [ 100*cm[i,i]/np.sum(cm[:,i]) if np.sum(cm[:,i]) else 0 for i in range(len(cm))]
    acsa =[ 100*(cm[0,0]/sum(cm[0,:])+cm[1,1]/sum(cm[1,:])+cm[2,2]/sum(cm[2,:]))/3]
    gm = [100*np.prod((cm[0,0]/sum(cm[0,:])*cm[1,1]/sum(cm[1,:])*cm[2,2]/sum(cm[2,:])))**(1/3)]
    r = class_acc + ppvs + acsa+gm
    # #%% plot
    print(r)
    # f = open(weight_path+'/model_history.json') 
    # h = json.load(f) 
    
    # plt.plot(h['acsa'])
    # plt.plot(h['val_acsa'])
    # plt.title('model acsa')
    # plt.ylabel('acsa')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig(weight_path+'acsa.png')
    # plt.show()
    
    # plt.plot(h['loss'])
    # plt.plot(h['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig(weight_path+'loss.png')
    # plt.show()
    
    # %% extract features 
    
    # with open(txt_file_path+'merged'+'.txt', 'r') as fr:
    #       all_files = fr.readlines()
         
    # model.load_weights(weight_path+'/checkpoint.h5')
    # intermediate_layer_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("Dense_1").output)
    # intermediate_layer_model.summary()
    
    # get_intermediate_output(preprocessed_image_path,all_files,intermediate_layer_model,intermedite_output_path)