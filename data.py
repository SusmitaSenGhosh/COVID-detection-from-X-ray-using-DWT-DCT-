import tensorflow as tf

import numpy as np
import os
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def process_image_file(filepath):
    img = cv2.imread(filepath)
    return img


class BalanceCovidDataset(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            data_dir,
            csv_file,
            is_training=True,    
            batch_size=8,
            input_shape=(224, 224),
            n_classes=3,
            num_channels=3,
            mapping={
                'normal': 0,
                'pneumonia': 1,
                'COVID-19': 2
            },
            shuffle=True,
    ):
        'Initialization'
        self.datadir = data_dir
        self.dataset = csv_file
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True
        self.n = 0
        
        datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        for l in self.dataset:
            datasets[l.split()[2]].append(l)
        self.datasets = [
            datasets['normal'], datasets['pneumonia'],
            datasets['COVID-19'],
        ]
        print(len(self.datasets[0]), len(self.datasets[1]), len(self.datasets[2]))
        #self.class_weights = [len(self.datasets[0])/len(self.dataset),len(self.datasets[1])/len(self.dataset),len(self.datasets[2])/len(self.dataset)]
        #self.individual_batch_size = [round(z*self.batch_size) for z in self.class_weights]  
       # print(self.individual_batch_size, self.__len__())                                                                                      
        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        self.n += 1
        batch_x, batch_y, weights = self.__getitem__(self.n)
        # Batch index

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0

        return batch_x, batch_y#, weights

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            #for v in self.datasets:
            np.random.shuffle(self.dataset)

    def __getitem__(self, idx):
        if idx == self.__len__()-1:
           # print('withinif',idx)            
            #batch_files = self.datasets[0][idx * self.individual_batch_size[0]:] + self.datasets[1][idx * self.individual_batch_size[1]:] + self.datasets[2][idx * self.individual_batch_size[2]:]
            batch_files = self.dataset[idx * self.batch_size:] 

        else:
           # print('withinelse',idx)            
            #batch_files = self.datasets[0][idx * self.individual_batch_size[0]:(idx + 1) * self.individual_batch_size[0]] + self.datasets[1][idx * self.individual_batch_size[1]:(idx + 1) * self.individual_batch_size[1]] + self.datasets[2][idx * self.individual_batch_size[2]:(idx + 1) * self.individual_batch_size[2]] 
            batch_files = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
            
        batch_x, batch_y = np.zeros(
            (len(batch_files), *self.input_shape,
             self.num_channels)), np.zeros(len(batch_files))
        
        for i in range(len(batch_files)):
            sample = batch_files[i].split()

            x = process_image_file(os.path.join(self.datadir, sample[1]))



            x = x.astype('float32') / 255.0
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

       # weights = np.take(class_weights, batch_y.astype('int64'))
        return batch_x, tf.keras.utils.to_categorical(batch_y, num_classes=self.n_classes)
