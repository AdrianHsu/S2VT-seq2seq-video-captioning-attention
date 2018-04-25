import numpy as np
import pandas as pd
import pickle
import os

from keras.preprocessing.text import text_to_word_sequence
from datasetBase import DatasetBase, DataObject

filters = '`","?!/.()'

class DatasetVal(DatasetBase):
    def __init__(self, data_dir, batch_size):
        super().__init__(data_dir, batch_size)
        self.feat_dir = self.data_dir + '/testing_data/feat/'
        self.json_filename = '/testing_label.json'
        self.corpus_dir = self.data_dir    

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

    def build_val_data_obj_list(self):
        corpus_path = self.corpus_dir + self.json_filename

        data_file = pd.read_json(corpus_path)
        max_size = 0
        for i in range(0, len(data_file['caption'])):

            myid = data_file['id'][i]
            path = self.feat_dir + myid + '.npy'
            mydat = np.load(path)
            str_list = data_file['caption'][i]
           
            tmp_list = []
            cap_len_list = [] 
            for j in range(0, len(str_list)):
                seq = text_to_word_sequence(str_list[j], filters=filters, 
                    lower=True, split=" ")
                tmp_list.append(seq)
                cap_len_list.append(len(seq) + 1) # added <EOS>

            obj = DataObject(path, myid, tmp_list, cap_len_list)
            self.dat_dict[myid] = mydat
            max_size += 1
            self.data_obj_list.append(obj)

        self.data_obj_list = np.array(self.data_obj_list)
        self.batch_max_size = max_size
        print('[Validation] total data size: ' + str(max_size))

    def next_batch(self): 
        
        # 1. sequential chosen, batch_size should be <= 100
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
        cap_batch = []
        id_batch = []
        cap_len = []
        for d in dat_list:
            img_batch.append(self.dat_dict[d.myid])
            id_batch.append(d.myid)
            cap, l = self.sample_one_caption(d.caption_list, d.cap_len_list) # randomly pick one
            cap = np.array(cap)
            cap_batch.append(cap)
            cap_len.append(l)
        cap_batch = self.captions_to_padded_sequences(cap_batch)

        return np.array(img_batch), np.array(cap_batch), np.array(cap_len), np.array(id_batch)

