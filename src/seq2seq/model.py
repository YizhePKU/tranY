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

        Args:
            input (max_sentence_len, batch_size): word ids of the sentences
            max_action_len: max number of actions the model is allowed to generated.

        Returns:
            logits (max_action_len, batch_size, action_vocab_size):
                model prediction in logits.
            actions (max_action_len, batch_size):
                model prediction in action ids.
        """
        max_sentence_len, batch_size = input.shape
        action_vocab_size = self.decoder.vocab_size

        encoder_output, encoder_hidden = self.encoder(input)
        assert encoder_output.shape == (
            max_sentence_len,
            batch_size,
            self.encoder.hidden_size,
        )

        # initialize decoder_hidden with the final value of encoder_hidden
        decoder_hidden = encoder_hidden
        logits = []
        actions = []
        # use SOA as the first prompt
        actions.append(
            torch.full(
                (batch_size,),
                self.special_tokens.index("[SOA]"),
                device=self.device,
            )
        )
        for i in range(max_action_len):
            _logits, decoder_hidden = self.decoder(
                actions[i].unsqueeze(0), decoder_hidden
            )
            logits.append(_logits)
            # TODO: is it OK to sample with argmax?
            _actions = torch.argmax(_logits, dim=1)
            actions.append(_actions)
        return (
            F.pad(torch.stack(logits), (0, max_action_len - len(logits))),
            F.pad(torch.stack(actions[1:]), (0, max_action_len - len(logits))),
        )
