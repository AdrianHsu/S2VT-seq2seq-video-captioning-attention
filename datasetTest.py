import numpy as np
import pickle
import os

from datasetBase import DatasetBase, DataObject

class DatasetTest(DatasetBase):
    def __init__(self, data_dir, test_dir, batch_size = 20):
        super().__init__(data_dir, batch_size)
        self.test_dir = test_dir # '/home/data/MLDS_hw2_1_data/testing_data'
        self.feat_dir = self.test_dir + '/feat/'
        self.batch_size = batch_size

        self.id_txt = '/id.txt'

    def load_tokenizer(self):
        # should be put in same folder!
        with open('word_index.pkl', 'rb') as handle:
            self.word_index = pickle.load(handle)
        with open('idx_to_word.pkl', 'rb') as handle:
            self.idx_to_word = pickle.load(handle)
        with open('word_counts.pkl', 'rb') as handle:
            self.word_counts = pickle.load(handle)

        self.vocab_num = len(self.word_counts) + 4
        return self.vocab_num

    def build_test_data_obj_list(self):

        txt = open(self.test_dir + self.id_txt, 'r')
        print('load txt: ' + self.test_dir + self.id_txt)
        max_size = 0

        for line in txt.readlines():

            myid = line.split('\n')[0]
            path = self.feat_dir + myid + '.npy'
            mydat = np.load(path)
           
            obj = DataObject(path, myid)
            self.dat_dict[myid] = mydat
            max_size += 1
            self.data_obj_list.append(obj)
        
        self.data_obj_list = np.array(self.data_obj_list)
        self.batch_max_size = max_size
        print('[Testing] total data size: ' + str(max_size))

    def next_batch(self): 
        
        # 1. sequential chosen
        current_index = self.batch_index
        max_size = self.batch_max_size
        if current_index + self.batch_size <= max_size:
            dat_list = self.data_obj_list[current_index:(current_index + self.batch_size)]
            self.batch_index += self.batch_size
        else:
            right = self.batch_size - (max_size - current_index)
            dat_list = np.append(self.data_obj_list[current_index:max_size], self.data_obj_list[0: right])
            self.batch_index = right
        
        img_batch = []
        id_batch = []
        for d in dat_list:
            img_batch.append(self.dat_dict[d.myid])
            id_batch.append(d.myid)
            
        return np.array(img_batch), np.array(id_batch)
