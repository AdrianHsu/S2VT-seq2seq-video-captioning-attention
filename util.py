import numpy as np
import time
from colors import *

np.random.seed(0)

special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
max_caption_len = 50
print_num = 10

def inv_sigmoid(num_epo):

    # 0.88 to 0.12 (-2.0 to 2.0)
    x = np.arange(-2.0, 2.0, (4.0/num_epo))
    y = 1/(1 + np.e**x)
    #y = np.ones(num_epo)
    print(y)
    return y

def linear_decay(num_epo):

    x = np.arange(0.0, 1.0, (1.0/num_epo))
    x = np.flip(x)
    return x

def dec_print_train(pred, cap_len, label, idx2word, batch_size, id_batch):
    
    i = np.random.randint(0, batch_size)
    eos_pred = max_caption_len - 1
    eos = cap_len[i] - 1
    for j in range(0, max_caption_len):
            if pred[i][j] == special_tokens['<EOS>']:
                eos_pred = j
                break
    
    pre = list( map (lambda x: idx2word[x] , pred[i][0:eos_pred])  )
    lab = list( map (lambda x: idx2word[x] , label[i][0:eos])  )
    print(color('\nid: ' + str(id_batch[i]) + '\nanswer: ' + str(lab) + '\nprediction: ' + str(pre), fg='yellow') )

def dec_print_val(pred, cap_len, label, idx2word, batch_size, id_batch):
     
    seq = []
    print_me = np.random.randint(batch_size, size=(1, print_num))
    for i in range(0, batch_size):
        eos_pred = max_caption_len - 1
        eos = cap_len[i] - 1
        for j in range(0, max_caption_len):
                if pred[i][j] == special_tokens['<EOS>']:
                    eos_pred = j
                    break
        myid = id_batch[i]
        pre = list( map (lambda x: idx2word[x] , pred[i][0:eos_pred])  )
        lab = list( map (lambda x: idx2word[x] , label[i][0:eos])  )
        pre_no_eos = list( map (lambda x: idx2word[x] , pred[i][0:(eos_pred)])  )
        sen = ' '.join([w for w in pre_no_eos])
        seq.append(sen)
        if i in print_me:
            # only print the "print_me"
            print(color('\nid: ' + str(myid) + '\nanswer: ' + str(lab) + '\nprediction: ' + str(pre), fg='green') )

    return seq

def dec_print_test(pred, idx2word, batch_size, id_batch):
    
    seq = []
    for i in range(0, batch_size):
        eos_pred = max_caption_len - 1
        for j in range(0, max_caption_len):
                if pred[i][j] == special_tokens['<EOS>']:
                    eos_pred = j
                    break
        pre = list( map (lambda x: idx2word[x] , pred[i][0:eos_pred])  )
        print(color('\nid: ' + str(id_batch[i]) + '\nlen: ' + str(eos_pred) + '\nprediction: ' + str(pre), fg='green') )
        pre_no_eos = list( map (lambda x: idx2word[x] , pred[i][0:(eos_pred)])  )
        sen = ' '.join([w for w in pre_no_eos])
        seq.append(sen)
    return seq
