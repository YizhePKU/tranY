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

        encoder_output, encoder_hidden = self.encoder(input)

        # initialize decoder_hidden with the final value of encoder_hidden
        decoder_hidden = encoder_hidden

        # use SOA as the first prompt
        init_action = torch.full(
            (batch_size,),
            self.special_tokens.index("[SOA]"),
            device=self.device,
        )

        logits = []
        actions = [init_action]
        for i in range(max_action_len):
            _logits, decoder_hidden = self.decoder(
                actions[i].unsqueeze(0), decoder_hidden
            )
            logits.append(_logits)
            # TODO: is it OK to sample with argmax?
            _actions = torch.argmax(_logits, dim=1)
            actions.append(_actions)
        return torch.stack(logits), torch.stack(actions[1:])
