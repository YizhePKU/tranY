import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, device):
        super(EncoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, device=device)

    def forward(self, input, hidden=None):
        """Feed input sentences to the encoder.

        Args:
            input (PackedSequence): ids of the input sentence.
            hidden: previous hidden state. defaults to zeros if not provided or None.

        Returns:
            output (PackedSequence): output of the encoder.
            hidden: new hidden state.
        """
        # apply embedding pointwise to the packed sequence
        embedded = torch.nn.utils.rnn.PackedSequence(
            self.embedding(input.data), input.batch_sizes, input.sorted_indices
        )
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden
