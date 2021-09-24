import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    """A standard seq2seq model.

    Input is fed to the LSTM encoder to get its final cell state. That cell state
    is fed to a linear layer to initialize the cell state of the LSTM decoder.
    Output is then sampled from decoder and selected with argmax.
    """

    def __init__(self, encoder, decoder, special_tokens, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.special_tokens = special_tokens
        self.device = device

        self.decoder_cell_init = nn.Linear(
            2 * encoder.hidden_size, decoder.hidden_size, device=device
        )

    def forward(self, input, label, max_recipe_len, teacher_forcing_p):
        """Feed words to the seq2seq model.

        Args:
            input (PackedSequence): ids of the input words.
            label (PackedSequence): ids of expected output actions, used for teacher forcing.
            max_recipe_len (int): maximum number of actions per recipe to generate.
            teacher_forcing_p (float): how often to use teacher forcing.
                0 = turn off teacher forcing
                1 = always use teacher forcing

        Returns:
            logits (max_recipe_len, batch_size, action_vocab_size): model predictions,
                including SOA and EOA.
        """
        assert 0 <= teacher_forcing_p <= 1
        assert type(input) == torch.nn.utils.rnn.PackedSequence
        batch_size = input.batch_sizes[0]

        # padded_label: (max_recipe_len x batch_size)
        padded_label = torch.nn.utils.rnn.pad_packed_sequence(
            label, total_length=max_recipe_len
        )[0]

        encoder_output, encoder_state = self.encoder(input)

        decoder_state = self._init_decoder_state(encoder_state)
        decoder_input = self._init_action(batch_size=batch_size)

        # As the model doesn't generate SOA, we need to add it to logits manually.
        SOA_logits = self._SOA_logits(
            batch_size=batch_size, action_vocab_size=self.decoder.vocab_size
        )

        logits = [SOA_logits]
        for i in range(1, max_recipe_len):
            _logits, decoder_state = self.decoder(
                decoder_input.unsqueeze(0), decoder_state
            )
            logits.append(_logits)

            if torch.rand(1) < teacher_forcing_p:
                # use teacher forcing
                decoder_input = padded_label[i]
            else:
                # don't use teacher forcing, sample with argmax instead
                decoder_input = torch.argmax(_logits, dim=1)
        return torch.stack(logits)

    def _init_decoder_state(self, encoder_state):
        """Initialize decoder state with encoder states."""
        encoder_hidden_state, encoder_cell_state = encoder_state
        permuted = encoder_cell_state.permute(1, 0, 2).reshape(
            -1, 2 * self.encoder.hidden_size
        )
        lineared = self.decoder_cell_init(permuted)
        tanh_ed = torch.tanh(lineared)
        decoder_cell_state = tanh_ed.view(1, -1, self.decoder.hidden_size)
        decoder_hidden_state = torch.zeros_like(decoder_cell_state)
        return decoder_hidden_state, decoder_cell_state

    def _init_action(self, batch_size):
        """Make an initial action to feed to the decoder."""
        # use SOA as the first prompt
        return torch.full(
            (batch_size,),
            self.special_tokens.index("[SOA]"),
            device=self.device,
        )

    def _SOA_logits(self, batch_size, action_vocab_size):
        """Make initial logits as part of the output."""
        return F.one_hot(self._init_action(batch_size), action_vocab_size).type(
            torch.float32
        )
