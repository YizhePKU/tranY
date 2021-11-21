from collections import namedtuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from asdl.recipe import Builder
from nltk.sem.logic import demoException
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def calculate_loss(logits, label):
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), label.flatten(), reduction="sum"
    )


def calculate_errors(logits, label):
    return torch.sum(torch.argmax(logits, dim=2) != label)


class TranY(pl.LightningModule):
    def __init__(
        self,
        encoder_vocab_size,
        encoder_embed_d,
        encoder_hidden_d,
        encoder_nlayers,
        decoder_vocab_size,
        decoder_embed_d,
        decoder_hidden_d,
        decoder_nlayers,
        dropout_p,
        learning_rate,
        **kwargs,
    ):
        # because we use dot-product cross-attention between decoder and encoder,
        # they need to have compatible hidden size
        assert decoder_hidden_d == 2 * encoder_hidden_d

        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            encoder_vocab_size,
            encoder_embed_d,
            encoder_hidden_d,
            encoder_nlayers,
            dropout_p,
        )
        self.init_decoder = InitDecoder(
            encoder_hidden_d, decoder_hidden_d, decoder_nlayers
        )
        self.decoder = Decoder(
            decoder_vocab_size,
            decoder_embed_d,
            decoder_hidden_d,
            decoder_nlayers,
            dropout_p,
        )
        self.merge_attention = MergeAttention(
            encoder_hidden_d, decoder_hidden_d, dropout_p
        )
        self.predictor = nn.Linear(decoder_hidden_d, decoder_vocab_size)

    def add_argparse_args(parser):
        group = parser.add_argument_group("TranY")
        group.add_argument("--encoder_embed_d", type=int, default=64)
        group.add_argument("--encoder_hidden_d", type=int, default=128)
        group.add_argument("--encoder_nlayers", type=int, default=1)
        group.add_argument("--decoder_embed_d", type=int, default=64)
        group.add_argument("--decoder_hidden_d", type=int, default=256)
        group.add_argument("--decoder_nlayers", type=int, default=1)
        group.add_argument("--dropout_p", type=int, default=0.3)
        group.add_argument("--learning_rate", type=float, default=1e-3)
        return parser

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def forward_teacher_forcing(self, sentence, label, sentence_length, label_length):
        batch_size = len(sentence_length)
        sentence_mask = (
            unpack(pack(sentence, sentence_length.cpu(), enforce_sorted=False))[0] > 0
        )
        encoder_output, _, encoder_state = self.encoder(sentence, sentence_length)

        # accumulate all attentional outputs
        decoder_att_outputs = []
        decoder_state = self.init_decoder(encoder_output, encoder_state, batch_size)
        prev_att_output = torch.zeros(
            (batch_size, self.hparams.decoder_hidden_d),
        ).type_as(encoder_output)
        # we generated max(label_length) - 1 actions, skipping SOA
        for t in range(max(label_length) - 1):
            decoder_output, decoder_state = self.decoder(
                label[t],
                prev_att_output,
                decoder_state,
            )
            # (batch_size, n_query = 1, key_d)
            query = prev_att_output.unsqueeze(1)
            # (batch_size, n_kv, key_d)
            key = value = encoder_output.transpose(0, 1)
            # (batch_size, n_query = 1, n_kv)
            mask = sentence_mask.transpose(0, 1).unsqueeze(1)
            # (batch_size, value_d)
            att = attention(query, key, value, mask).squeeze(1)
            prev_att_output = self.merge_attention(att, decoder_output)
            decoder_att_outputs.append(prev_att_output)
        logits = self.predictor(torch.stack(decoder_att_outputs))
        return logits

    def forward_beam_search(
        self,
        sentence,
        sentence_length,
        beam_width,
        result_count,
        action2id,
        id2action,
        grammar,
    ):
        assert sentence.shape[1] == 1
        assert sentence_length.shape[0] == 1
        sentence_mask = (
            unpack(pack(sentence, sentence_length.cpu(), enforce_sorted=False))[0] > 0
        )
        encoder_output, _, encoder_state = self.encoder(sentence, sentence_length)

        Node = namedtuple(
            "Node",
            ["score", "next_action", "prev_att_output", "decoder_state", "builder"],
        )
        # current candidate nodes
        nodes = []
        # collect `result_count` decode results
        results = []
        # the first action is always SOA
        init_action = action2id["[SOA]"]
        init_att_output = torch.zeros(
            (1, self.hparams.decoder_hidden_d),
        ).type_as(encoder_output)
        init_decoder_state = self.init_decoder(encoder_output, encoder_state, 1)
        init_builder = Builder(grammar)
        nodes.append(
            Node(0, init_action, init_att_output, init_decoder_state, init_builder)
        )
        while len(results) < result_count and len(nodes) > 0:
            print(
                [
                    f"Node(score={node.score}, next_action={id2action[node.next_action]})"
                    for node in nodes
                ]
            )
            new_nodes = []
            for node in nodes:
                score, next_action, prev_att_output, decoder_state, builder = node
                # ignore nodes that are too long
                if len(builder.history) > 40:
                    continue
                if builder.done:
                    # MR is done; ignore EOA and whatnot
                    results.append((score, builder.result))
                else:
                    # expand node
                    decoder_output, decoder_state = self.decoder(
                        torch.tensor(next_action).type_as(sentence).unsqueeze(0),
                        prev_att_output,
                        decoder_state,
                    )
                    # TODO: cleanup duplicate code
                    # (batch_size, n_query = 1, key_d)
                    query = prev_att_output.unsqueeze(1)
                    # (batch_size, n_kv, key_d)
                    key = value = encoder_output.transpose(0, 1)
                    # (batch_size, n_query = 1, n_kv)
                    mask = sentence_mask.transpose(0, 1).unsqueeze(1)
                    # (batch_size, value_d)
                    att = attention(query, key, value, mask).squeeze(1)
                    att_output = self.merge_attention(att, decoder_output)
                    logits = self.predictor(att_output)[0]
                    # select `beam_width` nodes and add them to the queue
                    if ("GenToken",) in builder.allowed_actions:
                        # select a GenToken action
                        # TODO: actually generate a non-empty token
                        if ("Reduce",) in builder.allowed_actions:
                            action = ("Reduce",)
                        else:
                            action = ("GenToken", "")
                        action_id = action2id[action]
                        new_builder = builder.apply_action(action)
                        new_nodes.append(
                            Node(
                                score, action_id, att_output, decoder_state, new_builder
                            ),
                        )
                    else:
                        # select an action from allowed actions
                        action_ids = []
                        for action in builder.allowed_actions:
                            try:
                                action_ids.append(action2id[action])
                            except KeyError:
                                # some actions never occurs in the training set; this is fine
                                pass
                        action_ids.sort(key=lambda id: logits[id])
                        # compute scores for all allowed actions
                        logits = logits.masked_fill(
                            torch.tensor(
                                [i not in action_ids for i in range(len(logits))]
                            ),
                            -1e9,
                        )
                        scores = F.log_softmax(logits, dim=-1)
                        for next_action in action_ids[-beam_width:]:
                            # add this to the queue
                            next_builder = builder.apply_action(id2action[next_action])
                            new_nodes.append(
                                Node(
                                    score + scores[next_action],
                                    next_action,
                                    att_output,
                                    decoder_state,
                                    next_builder,
                                ),
                            )
            # prune new nodes
            new_nodes.sort(key=lambda node: -node.score)
            nodes = new_nodes[-beam_width:]
        return results

    def training_step(self, batch, batch_idx):
        # train with teacher forcing
        sentence, label, sentence_length, label_length = batch
        batch_size = len(sentence_length)
        logits = self.forward_teacher_forcing(
            sentence, label, sentence_length, label_length
        )
        # skip SOA of the label
        loss = calculate_loss(logits, label[1:]) / batch_size
        errors = calculate_errors(logits, label[1:]) / batch_size
        return {
            "loss": loss,
            "errors": errors,
        }

    def training_epoch_end(self, outputs):
        loss = [d["loss"] for d in outputs]
        avg_loss = sum(loss) / len(loss)
        errors = [d["errors"] for d in outputs]
        avg_errors = sum(errors) / len(errors)
        self.log_dict(
            {
                "Train/loss": avg_loss,
                "Train/errors": avg_errors,
            }
        )

    def validation_step(self, batch, batch_idx):
        sentence, label, sentence_length, label_length = batch
        batch_size = len(sentence_length)
        logits = self.forward_teacher_forcing(
            sentence, label, sentence_length, label_length
        )
        loss = calculate_loss(logits, label[1:]) / batch_size
        errors = calculate_errors(logits, label[1:]) / batch_size
        return {
            "loss": loss,
            "errors": errors,
        }

    def validation_epoch_end(self, outputs):
        loss = [d["loss"] for d in outputs]
        avg_loss = sum(loss) / len(loss)
        errors = [d["errors"] for d in outputs]
        avg_errors = sum(errors) / len(errors)
        self.log_dict(
            {
                "Val/loss": avg_loss,
                "Val/errors": avg_errors,
            }
        )


class Encoder(nn.Module):
    """Standard LSTM encoder with dropout.

    Inputs:
        sentence (max_sentence_length, batch_size): input sentences.
        sentence_length (batch_size): length of non-padding input.
        state: (optional) previous encoder state; if None, initialize with zeros.

    Outputs:
        output (max_sentence_length, batch_size, hidden_d): the encoder output.
        output_length (batch_size): length of the encoder output.
        state: the final state of the encoder.
    """

    def __init__(
        self,
        vocab_size,
        embed_d,
        hidden_d,
        nlayers,
        dropout_p,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_d)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(
            embed_d,
            hidden_d,
            nlayers,
            bidirectional=True,
        )

    def forward(self, sentence, length, state=None):
        embedded = self.embedding(sentence)
        dropped = self.dropout(embedded)
        packed = pack(dropped, length.cpu(), enforce_sorted=False)
        output, state = self.lstm(packed, state)
        unpacked_output, length = unpack(output)
        return unpacked_output, length, state


class Decoder(nn.Module):
    """Standard LSTM decoder with dropout.

    Input:
        prev_action (batch_size): previous applied action.
        prev_att_output (batch_size, hidden_d): previous attentional decoder output.
        state: previous decoder state.

    Output:
        output: the decoder output.
        state: the new decoder state.
    """

    def __init__(self, vocab_size, embed_d, hidden_d, nlayers, dropout_p):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(vocab_size, embed_d)
        self.lstm = nn.LSTM(
            embed_d + hidden_d,
            hidden_d,
            nlayers,
        )

    def forward(self, prev_action, prev_att_output, state):
        # (batch_size, vocab_size)
        embed_prev_action = self.embedding(prev_action)

        # TODO: add parent feeding
        # (batch_size, vocab_size + hidden_d)
        lstm_input = torch.cat([embed_prev_action, prev_att_output], dim=-1)
        output, state = self.lstm(lstm_input.unsqueeze(0), state)
        output = output.squeeze(0)  # (batch_size, hidden_d)
        return output, state


class InitDecoder(nn.Module):
    """Initialize decoder state from encoder state and final encoder output."""

    def __init__(self, encoder_hidden_d, decoder_hidden_d, decoder_nlayers):
        super().__init__()
        self.decoder_hidden_d = decoder_hidden_d
        self.decoder_nlayers = decoder_nlayers
        self.linear = nn.Linear(2 * encoder_hidden_d, decoder_hidden_d)

    def forward(self, encoder_output, encoder_state, batch_size):
        decoder_hidden = torch.zeros(
            (self.decoder_nlayers, batch_size, self.decoder_hidden_d)
        ).type_as(encoder_output)
        encoder_hidden, encoder_cell = encoder_state
        encoder_cell = encoder_cell.transpose(0, 1).reshape(batch_size, -1)
        decoder_cell = self.linear(encoder_cell).unsqueeze(0)
        return decoder_hidden, decoder_cell


def attention(query, key, value, mask=None):
    """Compute scaled dot-product attention.

    `query` is a tensor of shape (*, n_query, key_d).
    `key` is a tensor of shape (*, n_kv, key_d).
    `value` is a tensor of shape (*, n_kv, value_d).
    `mask` is a tensor of shape (*, n_query, n_kv). If None, no mask is applied.

    Return a tensor of shape (*, n_query, value_d).
    """
    key_d = query.shape[-1]
    factor = torch.sqrt(torch.tensor(key_d))
    # (*, n_query, n_kv)
    weights = (query @ key.transpose(-1, -2)) / factor
    if mask is not None:
        weights = weights.masked_fill(mask == 0, -1e9)
    # (*, n_query, n_kv)
    prob = F.softmax(weights, dim=-1)
    # (*, n_query, value_d)
    att = prob @ value
    return att


class MergeAttention(nn.Module):
    """Merge attention with decoder output.

    Transformer uses residue connection + LayerNorm, but tranX uses dropout + Linear + tanh.
    """

    def __init__(self, encoder_hidden_d, decoder_hidden_d, dropout_p):
        super().__init__()
        self.linear = nn.Linear(
            2 * encoder_hidden_d + decoder_hidden_d, decoder_hidden_d
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, att, decoder_output):
        att_output = torch.cat([att, decoder_output], dim=-1)
        att_output = self.dropout(att_output)
        att_output = torch.tanh(self.linear(att_output))
        return att_output
