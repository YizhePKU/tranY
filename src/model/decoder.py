import torch
import torch.nn as nn


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
        """Feed recipes to the decoder.

        Args:
            input (seq_length x batch_size): ids of the input recipes.
            hidden: previous hidden state. defaults to zeros if not provided or None.

        Returns:
            logits (batch_size x vocab_size): prediction for the next action.
            hidden: new hidden state.
        """
        action_embeddings = self.embedding(input)
        output, hidden = self.lstm(action_embeddings, hidden)
        logits = self.out(output[-1])
        return logits, hidden
