import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, special_tokens, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.special_tokens = special_tokens
        self.device = device

    def forward(self, input, max_action_len):
        """Feed sentences to the seq2seq model.

        Decoder outputs are sampled with argmax and fed back to the decoder.

        Args:
            input (PackedSequence): word ids of the sentences
            max_action_len (int): number of actions the model is allowed to generate.

        Returns:
            logits (max_action_len, batch_size, action_vocab_size): model predictions.
            actions (max_action_len, batch_size): model prediction in action ids,
                including EOA but not SOA.
        """
        assert type(input) == torch.nn.utils.rnn.PackedSequence
        batch_size = input.batch_sizes[0]

        encoder_output, encoder_state = self.encoder(input)

        # initialize decoder_state from final encoder_state
        decoder_state = _init_decoder_state(encoder_state)

        # use SOA as the first prompt
        init_action = torch.full(
            (batch_size,),
            self.special_tokens.index("[SOA]"),
            device=self.device,
        )

        logits = []
        actions = [init_action]
        for i in range(max_action_len):
            _logits, decoder_state = self.decoder(
                actions[i].unsqueeze(0), decoder_state
            )
            logits.append(_logits)
            # TODO: is it OK to sample with argmax?
            _actions = torch.argmax(_logits, dim=1)
            actions.append(_actions)
        return torch.stack(logits), torch.stack(actions[1:])


def _init_decoder_state(encoder_state):
    # Assuming encoder.hidden_size * 2 == decoder.hidden_size,
    # we can just concat the two encoder state to get the decoder state
    hidden_state, cell_state = encoder_state
    _, batch_size, encoder_hidden_size = hidden_state.shape
    return (
        hidden_state.permute(1, 0, 2).reshape(1, batch_size, 2 * encoder_hidden_size),
        cell_state.permute(1, 0, 2).reshape(1, batch_size, 2 * encoder_hidden_size),
    )
