import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.events import add_event


def train(
    encoder,
    decoder,
    input_tensor,
    output_tensor,
    encoder_optimizer,
    decoder_optimizer,
    decoder_init_action,
    EOA,
    n_epochs,
    max_sentence_length,
    max_action_length,
):
    """Train a seq2seq model.

    Args:
        encoder (nn.Module): encoder of the seq2seq model.
        decoder (nn.Module): decoder of the seq2seq model.
        input_tensor (max_sentence_length x batch_size):
            tensor that represents input sentences.
        output_tensor (max_action_length x batch_size):
            tensor that represents output action sequences.
        encoder_optimizer (nn.optim.Optimizer): optimizer for the encoder.
        decoder_optimizer (nn.optim.Optimizer): optimizer for the decoder.
        decoder_init_action (int): the initial action for the decoding process.
        EOA (int): index that represents end_of_sentence for actions.
        n_epochs (int): number of epochs to train.
        max_sentence_length (int): max length of the sentence the model should process.
        max_action_length (int): max length of the action sequence the model should generate.
    """
    batch_size = input_tensor.shape[1]
    assert batch_size == output_tensor.shape[1]

    for i in range(n_epochs):
        add_event(
            {
                "event": "TrainingEpochStart",
                "epoch": i,
            }
        )

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # feed input_tensor to encoder
        # encoder_outputs: (input_length x batch_size x hidden_size)
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # generate model-predicted output one at a time
        # initialize decoder_hidden with the final value of encoder_hidden
        decoder_logits = []
        decoder_hidden = encoder_hidden
        # last_action: (batch_size)
        last_action_idx = torch.full((batch_size,), decoder_init_action, device=decoder.device)
        while last_action_idx != EOA and len(decoder_logits) < max_action_length:
            # treat last_action_idx as an action sequence of length 1
            # logits: (batch_size x action_vocab_size)
            logits, decoder_hidden = decoder(
                last_action_idx.unsqueeze(0), decoder_hidden
            )
            decoder_logits.append(logits)
            # TODO: is it OK to sample with argmax?
            last_action_idx = torch.argmax(logits, dim=1)

        # calculate loss
        # decoder_logits: (action_length x batch_size x action_vocab_size)
        # output_tensor: (max_action_length x batch_size)
        loss = 0
        for i in range(len(output_tensor)):
            # if decoder_logits is not long enough, reuse last logits(which should be EOA)
            if i < len(decoder_logits):
                logits = decoder_logits[i]
            loss += F.cross_entropy(logits, output_tensor[i])
            if output_tensor[i] == EOA:
                break

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        add_event(
            {
                "event": "TrainingEpochEnd",
                "loss": loss.item(),
            }
        )
