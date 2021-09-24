import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, special_tokens, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.special_tokens = special_tokens
        self.device = device

    def forward(self, input, label, max_action_len, teacher_forcing_p):
        """Feed sentences to the seq2seq model.

        Decoder outputs are sampled with argmax.

        Args:
            input (PackedSequence): ids of the input words.
            label (PackedSequence): ids of expected output actions, used for teacher forcing.
            max_action_len (int): number of actions the model is allowed to generate.
            teacher_forcing_p (float): probability to use teacher forcing at each time step.
                0 = turn off teacher forcing
                1 = always use teacher forcing

        Returns:
            logits (max_action_len, batch_size, action_vocab_size): model predictions,
                including SOA and EOA.
        """
        assert 0 <= teacher_forcing_p <= 1
        assert type(input) == torch.nn.utils.rnn.PackedSequence
        batch_size = input.batch_sizes[0]

        # padded_label: (max_action_len x batch_size)
        padded_label = torch.nn.utils.rnn.pad_packed_sequence(
            label, total_length=cfg.max_action_len
        )[0]

        encoder_output, encoder_state = self.encoder(input)

        decoder_state = self._init_decoder_state(encoder_state)
        decoder_input = self._init_action(batch_size=batch_size)
        init_logits = self._init_logits(
            batch_size=batch_size, action_vocab_size=self.decoder.vocab_size
        )

        logits = [init_logits]
        for i in range(1, max_action_len):
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
        """Initialize decoder_state from final encoder_state."""
        # Assuming encoder.hidden_size * 2 == decoder.hidden_size,
        # we can just concat the two encoder state to get the decoder state
        hidden_state, cell_state = encoder_state
        _, batch_size, encoder_hidden_size = hidden_state.shape
        return (
            hidden_state.permute(1, 0, 2).reshape(
                1, batch_size, 2 * encoder_hidden_size
            ),
            cell_state.permute(1, 0, 2).reshape(1, batch_size, 2 * encoder_hidden_size),
        )

    def _init_action(self, batch_size):
        """Make an initial action to feed to the decoder."""
        # use SOA as the first prompt
        return torch.full(
            (batch_size,),
            self.special_tokens.index("[SOA]"),
            device=self.device,
        )

    def _init_logits(self, batch_size, action_vocab_size):
        """Make initial logits as part of the output."""
        return F.one_hot(self._init_action(batch_size), action_vocab_size).type(
            torch.float32
        )
