'''
Calculate Spearman Correlation 
'''


import torch
import numpy as np
from scipy.stats import spearmanr


def txt2list(txt_path):
    '''
    Read a txt file and return a list
    Arguments:
        txt_path (str): path to save the text file
    '''
    output = []
    f = open(txt_path, 'r')
    for line in f.readlines():
        output.append(line.replace('\n', ''))
    f.close()
    return output


def get_spearman_cor(ws_txt, model, word2idx, cuda=False):
    lines = txt2list(ws_txt)
    human_sims = []
    pred_sims = []
    for line in lines:
        w1, w2, human_sim = line.split('\t')
        if w1 not in word2idx or w2 not in word2idx:
            continue
        t1 = torch.tensor(word2idx[w1])
        t2 = torch.tensor(word2idx[w2])
        if cuda:
            t1 = t1.cuda()
            t2 = t2.cuda()
        pred_sim = model.get_word_sim(t1, t2).item()
        human_sims.append(float(human_sim))
        pred_sims.append(pred_sim)
    sc = spearmanr(human_sims, pred_sims)
    print('Spearman Correlation: {}'.format(sc))


if __name__ == '__main__':
    read_ws_data('./data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt')
