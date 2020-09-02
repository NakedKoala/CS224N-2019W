#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, embed_size, num_filter, kernel_size, padding):
        super().__init__()
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.conv_layer = nn.Conv1d(in_channels=embed_size, out_channels=num_filter, \
                                    kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # @param x: x_reshape of shape (batch_size,  embed_size, max_word_length)
        # @returns x_conv_out of shape (batch_size, num_filter)
        batch_size, embed_size, max_word_length  = x.shape 
        x_conv = self.conv_layer(x)
        # assert(x_conv.shape == (batch_size, self.num_filter, max_word_length - self.kernel_size + 1 + 2))
        x_conv_out =  torch.max(nn.functional.relu(x_conv), dim=-1)[0]
        # import pdb 
        # pdb.set_trace()
        # assert(x_conv_out.shape == (batch_size, self.num_filter))
        return x_conv_out

    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g




    ### END YOUR CODE

