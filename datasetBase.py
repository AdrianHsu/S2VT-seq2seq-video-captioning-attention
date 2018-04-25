import numpy as np


np.random.seed(0)

word_min_counts_threshold = 3
max_caption_len = 50

special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}

class DataObject:
    def __init__(self, path, myid, caption_list = None, cap_len_list = None):
        self.path = path
        self.myid = myid
        self.caption_list = caption_list # no EOS, e.g. ['I', 'love', 'you']
        self.cap_len_list = cap_len_list # EOS added, e.g. 4
 
class DatasetBase:
    def __init__(self, data_dir, batch_size):
        self.data_obj_list = []
        self.word_min_counts_threshold = word_min_counts_threshold
        self.vocab_num = 0
        self.word_counts = {}
        self.word_index = {}
        self.idx_to_word = {}
        self.dat_dict = {}
        self.data_dir = data_dir
        self.batch_max_size = 0
        self.batch_size = batch_size
        self.batch_index = 0

    def sample_one_caption(self, captions, cap_len, is_rand=True):

        assert len(captions) == len(cap_len)
        if is_rand:
            r = np.random.randint(0, len(captions))
        else:
            r = 0
        return captions[r], cap_len[r]

    def captions_to_padded_sequences(self, captions, maxlen=max_caption_len):

        res = []
        for cap in captions:
            l = []
            for word in cap:
                if word in self.word_counts:
                    l.append(self.word_index[word])
                else:
                    l.append(special_tokens['<UNK>'])
            l.append(special_tokens['<EOS>']) # add EOS here!
            pad = special_tokens['<PAD>']
            l += [ pad ] * (maxlen - len(l))
            res.append(l)
        return res

