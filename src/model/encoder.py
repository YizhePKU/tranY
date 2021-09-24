import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout_p, device):
        super(EncoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, bidirectional=True, device=device
        )

    def forward(self, input):
        """Feed words to the encoder.

        Args:
            input (PackedSequence): ids of the input words.

        Returns:
            encoding (PackedSequence x embedding_dim): encodings of the input words.
            hidden: new hidden state.
        """
        # apply embedding+dropout pointwise to the packed sequence
        embedded = self.embedding(input.data)
        dropped = self.dropout(embedded)
        packed = torch.nn.utils.rnn.PackedSequence(
            dropped, input.batch_sizes, input.sorted_indices
        )
        encoding, hidden = self.lstm(packed)
        return encoding, hidden
