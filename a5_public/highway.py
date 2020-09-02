#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, embed_size, dropout_rate):
        super().__init__()
        self.embed_size = embed_size
        self.gate_layer = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.proj_layer = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        #@param x: x is x_conv_out of shape (batch_size, embed_size)
        #@returs x_word_embd: x is the embeddings for batch of word. Shape: (batch, embed_size)

        batch_size = x.shape[0]
        # assert(x.shape == (batch_size, self.embed_size))
        x_gate = torch.sigmoid(self.gate_layer(x))
       
        x_proj = nn.functional.relu(self.proj_layer(x))
        # assert(x_gate.shape == (batch_size, self.embed_size))
        # assert(x_proj.shape == (batch_size, self.embed_size))

        x_highway = x_gate * x_proj + (1 - x_gate) * x 
        x_word_emb = self.dropout(x_highway)
        # assert(x_word_emb.shape == (batch_size, self.embed_size))
        return x_word_emb

    ### END YOUR CODE

