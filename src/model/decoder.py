import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, device):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, device=device)
        self.out = nn.Linear(hidden_size, vocab_size, device=device)

    def forward(self, input, hidden):
        """Feed actions to the decoder.

        Args:
            input (seq_length x batch_size): ids of the input actions.
            hidden: previous hidden state. defaults to zeros if not provided or None.

        Returns:
            logits (batch_size x vocab_size): prediction for the next action.
            hidden: new hidden state.
        """
        # embedded: (seq_length x batch_size x embedding_dim)
        embedded = self.embedding(input)
        # embedded_relu: (seq_length x batch_size x embedding_dim)
        embedded_relu = F.relu(embedded)
        embedded_relu_dropped = self.dropout(embedded_relu)
        # output: (seq_length x batch_size x hidden_size)
        output, hidden = self.lstm(embedded_relu_dropped, hidden)
        # logits: (batch_size x vocab_size)
        logits = self.out(output[-1])
        return logits, hidden
