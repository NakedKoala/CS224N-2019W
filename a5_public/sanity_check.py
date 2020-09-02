#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py 1e
    sanity_check.py 1f
    sanity_check.py 1g
    sanity_check.py 1h
    sanity_check.py 2a
    sanity_check.py 2b
    sanity_check.py 2c
"""
import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, batch_iter, read_corpus
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from highway import Highway
from cnn import CNN
from nmt_model import NMT


import torch
import torch.nn as nn
import torch.nn.utils

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 4
DROPOUT_RATE = 0.0

class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        # import pdb 
        # pdb.set_trace()
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_pad = self.char2id['<pad>']
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

def question_1g_sanity_check(): 

    x_reshape = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341, 0.4901],
                                [0.8964, 0.4556, 0.6323, 0.3489, 0.4017, 0.0223, 0.1689],
                                [0.2939, 0.5185, 0.6977, 0.8000, 0.1610, 0.2823, 0.6816],
                                [0.9152, 0.3971, 0.8742, 0.4194, 0.5529, 0.9527, 0.0362]],
                            [[0.1852, 0.3734, 0.3051, 0.9320, 0.1759, 0.2698, 0.1507],
                            [0.0317, 0.2081, 0.9298, 0.7231, 0.7423, 0.5263, 0.2437],
                            [0.5846, 0.0332, 0.1387, 0.2422, 0.8155, 0.7932, 0.2783],
                            [0.4820, 0.8198, 0.9971, 0.6984, 0.5675, 0.8352, 0.2056]]])

    conv_w = nn.Parameter(torch.tensor([[[ 0.0417, -0.1734, -0.1550, -0.1155,  0.1012],
                                        [ 0.0899, -0.1325,  0.0676,  0.1228, -0.0282],
                                        [ 0.0085,  0.0518,  0.1387,  0.2147, -0.1723],
                                        [-0.0819,  0.0879,  0.1853,  0.1946,  0.1973]],
                                        [[ 0.0445, -0.1944,  0.0206, -0.1399, -0.2084],
                                        [ 0.1987,  0.1700, -0.2231,  0.0419, -0.0377],
                                        [-0.0368, -0.1024,  0.0860, -0.1324,  0.0820],
                                        [ 0.1131,  0.1601,  0.0836, -0.2213, -0.1451]],
                                        [[ 0.1117,  0.0468, -0.1744, -0.1288,  0.2104],
                                        [ 0.1507, -0.0975, -0.0563, -0.2130, -0.0040],
                                        [-0.1684, -0.1725, -0.0123,  0.0336, -0.0916],
                                        [ 0.1327, -0.1361,  0.2029,  0.1532, -0.1886]]]))

    conv_b = nn.Parameter(torch.tensor([-0.0557,  0.0101,  0.0326]))

    cnn = CNN(embed_size=4, num_filter=3, kernel_size=5, padding=1)
    cnn.conv_layer.weight = conv_w 
    cnn.conv_layer.bias = conv_b
    actual_x_conv_out = cnn(x_reshape)
    target_x_conv_out = torch.tensor([[0.5537, 0.1360, 0.2047],
                                      [0.5660, 0.2976, 0.1589]])
    assert(np.allclose(target_x_conv_out, actual_x_conv_out.detach().numpy(), 1e-3))

    print("Test passed")


def question_1f_sanity_check(): 
    x = torch.tensor([[0.4963, 0.7682],
                    [0.0885, 0.1320],
                    [0.3074, 0.6341]])
    gate_layer_w = nn.Parameter(torch.tensor([[-0.0140,  0.5607],
                                              [-0.0628,  0.1871]]))
    gate_layer_b = nn.Parameter(torch.tensor([-0.2137, -0.1390]))

    proj_layer_w = nn.Parameter(torch.tensor([[-0.6755, -0.4683],
                                            [-0.2915,  0.0262]]))
    proj_layer_b = nn.Parameter(torch.tensor([0.2795, 0.4243]))

    highway = Highway(embed_size=2, dropout_rate=0)
    with torch.no_grad():
        highway.gate_layer.weight = gate_layer_w
        highway.gate_layer.bias = gate_layer_b
        highway.proj_layer.weight = proj_layer_w
        highway.proj_layer.bias = proj_layer_b

    actual_x_word_embed = highway(x)

    target_x_word_embed = np.array([[0.2222, 0.5371],
                                    [0.1208, 0.2589],
                                    [0.1432, 0.4955]])
   
    assert(np.allclose(target_x_word_embed, actual_x_word_embed.detach().numpy(),1e-3))
    print("Test passed !")



def question_1e_sanity_check():
    """ Sanity check for to_input_tensor_char() function.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1e: To Input Tensor Char")
    print ("-"*80)
    vocabEntry = VocabEntry()

    print("Running test on a list of sentences")
    sentences = [['Human', ':', 'What', 'do', 'we', 'want', '?'], ['Computer', ':', 'Natural', 'language', 'processing', '!'], ['Human', ':', 'When', 'do', 'we', 'want', 'it', '?'], ['Computer', ':', 'When', 'do', 'we', 'want', 'what', '?']]
    sentence_length = 8
    BATCH_SIZE = 4
    word_length = 12
    output = vocabEntry.to_input_tensor_char(sentences, 'cpu')
    output_expected_size = [sentence_length, BATCH_SIZE, word_length]
    assert list(output.size()) == output_expected_size, "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))

    print("Sanity Check Passed for Question 1e: To Input Tensor Char!")
    print("-"*80)

def question_1h_sanity_check(model):
    """ Sanity check for model_embeddings.py
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1h: Model Embedding")
    print ("-"*80)
    sentence_length = 10
    max_word_length = 21
    inpt = torch.zeros(sentence_length, BATCH_SIZE, max_word_length, dtype=torch.long)
    ME_source = model.model_embeddings_source
    output = ME_source.forward(inpt)
    output_expected_size = [sentence_length, BATCH_SIZE, EMBED_SIZE]
    assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    print("Sanity Check Passed for Question 1h: Model Embedding!")
    print("-"*80)

def question_2a_sanity_check(decoder, char_vocab):
    """ Sanity check for CharDecoder.forward()
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 2a: CharDecoder.forward()")
    print ("-"*80)
    sequence_length = 4
    inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
    logits, (dec_hidden1, dec_hidden2) = decoder.forward(inpt)
    logits_expected_size = [sequence_length, BATCH_SIZE, len(char_vocab.char2id)]
    dec_hidden_expected_size = [1, BATCH_SIZE, HIDDEN_SIZE]
    assert(list(logits.size()) == logits_expected_size), "Logits shape is incorrect:\n it should be {} but is:\n{}".format(logits_expected_size, list(logits.size()))
    assert(list(dec_hidden1.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(dec_hidden_expected_size, list(dec_hidden1.size()))
    assert(list(dec_hidden2.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(dec_hidden_expected_size, list(dec_hidden2.size()))
    print("Sanity Check Passed for Question 2a: CharDecoder.forward()!")
    print("-"*80)

def question_2b_sanity_check(decoder):
    """ Sanity check for CharDecoder.train_forward()
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 2b: CharDecoder.train_forward()")
    print ("-"*80)
    sequence_length = 4
    inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
    loss = decoder.train_forward(inpt)
    assert(list(loss.size()) == []), "Loss should be a scalar but its shape is: {}".format(list(loss.size()))
    print("Sanity Check Passed for Question 2b: CharDecoder.train_forward()!")
    print("-"*80)

def question_2c_sanity_check(decoder):
    """ Sanity check for CharDecoder.decode_greedy()
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 2c: CharDecoder.decode_greedy()")
    print ("-"*80)
    sequence_length = 4
    inpt = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float)
    initialStates = (inpt, inpt)
    device = decoder.char_output_projection.weight.device
    decodedWords = decoder.decode_greedy(initialStates, device)
    assert(len(decodedWords) == BATCH_SIZE), "Length of decodedWords should be {} but is: {}".format(BATCH_SIZE, len(decodedWords))
    print("Sanity Check Passed for Question 2c: CharDecoder.decode_greedy()!")
    print("-"*80)

def main():
    """ Main func.
    """
    args = docopt(__doc__)
    

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    model = NMT(
        word_embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    char_vocab = DummyVocab()

    # Initialize CharDecoder
    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)

    if args['1e']:
        question_1e_sanity_check()
    elif args['1f']:
        question_1f_sanity_check()
    elif args['1h']:
        question_1h_sanity_check(model)
    elif args['1g']:
        question_1g_sanity_check()
    elif args['2a']:
        question_2a_sanity_check(decoder, char_vocab)
    elif args['2b']:
        question_2b_sanity_check(decoder)
    elif args['2c']:
        question_2c_sanity_check(decoder)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
