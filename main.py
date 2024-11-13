import os
import torch
import torch_npu
import torch.utils.data as Data
from model import Transformers
from data import MyDataSet
import torch.nn as nn

model = Transformers()

device = torch.npu.set_device(7)
# Padding Should be Zero
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab)
 
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
target_vocab_size = len(tgt_vocab)
 
src_len = 5  # enc_input max sequence length
tgt_len = 6  # dec_input(=dec_output) max sequence length
 
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
 
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
 
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
 
 
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
dec_out = dec_outputs.contiguous().view(-1)

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
cirection = nn.CrossEntropyLoss()
for enc_inputs, dec_inputs, dec_outputs in loader:
    output, enc_self_attn, dec_self_attn, dec_enc_attn = model(enc_inputs, dec_inputs)
    # loss  = cirection(output, dec_outputs)
    loss  = cirection(output, dec_out)
    print("loss: ", loss)
