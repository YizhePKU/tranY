import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pointer import PointerNet
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = nn.Linear(self.decoder.hidden_size, self.decoder.action_size, device=self.encoder.device)
        
    def forward(self, src, tgt, src_mask, tgt_mask, src_lengths, tgt_lengths):
        """Take in and process masked src and target sequences."""
        encoder_output, encoder_hidden = self.encoder(src, src_lengths)
        out, hidden, preout = self.decoder(tgt, encoder_output, encoder_hidden, src_mask, tgt_mask)
        pred = self.generator(preout)
        return pred


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, nlayers, device, dp=0.3, bidirect=True):
        super(EncoderLSTM, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.brnn = bidirect
        self.device = device
        self.dp = nn.Dropout(dp)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, nlayers, batch_first=True,
                            bidirectional=bidirect, device=device)
        
    def forward(self, inputs, lengths, hidden=None):
        inputs = self.embedding(inputs)
        inputs = self.dp(inputs)
        packed = pack(inputs, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed, hidden)
        output, _ = unpack(output, batch_first=True)
        return output, hidden


class DecoderLSTM(nn.Module):
    def __init__(self, action_size, encoder_dim, embedding_dim, hidden_size, nlayers, device, dp=0.3):
        super(DecoderLSTM, self).__init__()
        self.device = device
        self.action_size = action_size
        self.encoder_dim = encoder_dim
        self.embedding_size = embedding_dim
        self.hidden_size = hidden_size
        self.device = device
        self.dp = nn.Dropout(dp)
        # TODO: init decoder with encoder state
        # self.bridge1 = nn.Linear(encoder_dim, hidden_size)
        # self.bridge2 = nn.Linear(encoder_dim, hidden_size)
        self.embedding = nn.Embedding(action_size, embedding_dim, device=device)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, nlayers, batch_first=True, device=device)
        self.out = nn.Linear(encoder_dim + hidden_size, hidden_size, device=device)

    def attention(self, query, key, value, mask):
        """
            Perform a standand attention.
            Inputs:
                query: Tensor of (B, 1, d1)
                key: Tensor of (B, L, d1)
                value: Tensor of (B, L, d2)
                mask: Boolean Tensor of (B, L)
                Note that d1 may equal to d2.
            Outputs:
                context: Tensor of attention summed value (B, d2)
                attn_probs: Tensor of (B, L)
        """
        attn_probs = (query * key).sum(-1)     #(B, L)
        attn_probs.masked_fill_(mask, -np.inf)
        attn_probs = nn.Softmax(dim=-1)(attn_probs)
        context = torch.sum(attn_probs.unsqueeze(-1) * value, dim=1).unsqueeze(1)
        return context, attn_probs

    def forward_step(self, prev_embed, pre_output, pre_hidden, encoder_hidden, src_mask):
        """
            Perform a single decoder step (1 word)
            Inputs:
                prev_embed: shape (B, 1, D)
        """
        # compute context vector using attention mechanism
        query = pre_output
        context, attn_probs = self.attention(
            query=query, key=encoder_hidden,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, pre_output], dim=2)
        output, hidden = self.lstm(rnn_input, pre_hidden)
        pre_output = torch.cat([output, context], dim=2)
        pre_output = self.dp(pre_output)
        pre_output = nn.Tanh()(self.out(pre_output))
        return output, hidden, pre_output

    def forward(self, targets, encoder_hidden, encoder_final, 
                src_mask, tgt_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = tgt_mask.size(-1)
        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        B, L = targets.shape
        H = self.hidden_size
        targets = self.embedding(targets)
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        pre_output = torch.autograd.Variable(torch.zeros(B, 1, H).to(self.device), requires_grad=False)
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = targets[:, i, :].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, pre_output, hidden, encoder_hidden, src_mask)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)
        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]
    
    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""
        return None
        # if encoder_final is None:
        #     return None  # start with zeros
        # return (torch.tanh(self.bridge1(encoder_final[0])),
        #          torch.tanh(self.bridge2(encoder_final[1])))


class CopyDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, device, dp=0.3):
        super(CopyDecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, device=device)
        self.pointer = PointerNet(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size, device=device)
        self.dp = nn.Dropout(0.2)

    def forward(self, inputs, src_input):
        """
        LSTM Decoder with Copy Mechanism
        Args:
            input (batch_size x decoder_length): indices of the target sentence.
            src_input: (batch_size x encoder_length x vocab_size)
        Returns:
            logits (batch_size x decoder_length x vocab_size): logits for the next word.
        """
        embedded = self.embedding(inputs)
        embedded = nn.ReLU(embedded)
        outputs, __ = self.lstm(embedded)
        gens = self.out(outputs)
        pass