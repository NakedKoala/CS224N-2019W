#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import numpy as np

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.char_embedding_size = char_embedding_size
        self.hidden_size = hidden_size
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)
        self.crit = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        # Inputs: input, (h_0, c_0)
        # input -> (seq_len, batch, input_size)
        # import pdb 
        # pdb.set_trace()
        seq_len, batch_size = input.shape
        input_embedding = self.decoderCharEmb(input.reshape(-1)).reshape(seq_len, batch_size, self.char_embedding_size)

        if dec_hidden:
            o, dec_hidden = self.charDecoder(input_embedding, dec_hidden)
        else:
            o, dec_hidden = self.charDecoder(input_embedding)

        # assert(o.shape == (seq_len, batch_size, self.hidden_size))
        scores = self.char_output_projection(o)
        # assert(scores.shape == (seq_len, batch_size, len(self.target_vocab.char2id)))

        return scores, dec_hidden

        



        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        input_sequence = char_sequence[:-1,:]
        target_sequence = char_sequence[1:,:]
        scores, dec_hidden = self.forward(input_sequence, dec_hidden)
        #(length, batch_size, self.vocab_size)
        
        loss = self.crit(scores.reshape(-1, len(self.target_vocab.char2id)),  target_sequence.reshape(-1))
        # import pdb 
        # pdb.set_trace()

        return loss 

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        ### END YOUR CODE
        # import pdb 
        # pdb.set_trace()
        _, batch_size, _ = initialStates[0].shape
        dec_hidden = initialStates
        output_word = []
        current_char = torch.empty(batch_size, device=device).fill_(self.target_vocab.char2id['{']).reshape(1,-1).long()

        for t in range(max_length):
            # input -> (length, batch_size)
            # scores -> (length, batch_size, self.vocab_size)
            # import pdb 
            # pdb.set_trace()
            scores, dec_hidden =  self.forward(current_char, dec_hidden)
            p = nn.functional.softmax(scores, dim=-1)
            indices =  torch.argmax(p, dim=-1).detach().cpu().numpy().reshape(-1).tolist()
            to_append = [ self.target_vocab.id2char[idx] for idx in indices]
            output_word.append(to_append)
            current_char = torch.tensor(indices, device=device).reshape(1,-1)
            # assert(current_char.shape == (1, batch_size))

        # import pdb 
        # pdb.set_trace()
        output_word = np.stack(np.array(output_word), axis=-1)
        # assert(output_word.shape == (batch_size, max_length))
        output_word = np.apply_along_axis(lambda x: "".join(x), 1, output_word)
        # assert(len(output_word) == batch_size)
        # import pdb 
        # pdb.set_trace()
        output_word = [ item[: item.find('}') ]  if item.find('}') != -1 else item for item in output_word]
        
        return output_word
       



