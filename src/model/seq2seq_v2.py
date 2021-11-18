import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class EncoderDecoder(nn.Module):
    """A standard Encoder-Decoder architecture.

    Inputs:
        input (N1, batch_size): the input words.
        label (N2, batch_size): the target label.
            This is used for teacher forcing.
        input_length (batch_size): length of non-padding input.
        label_length (batch_size): length of non-padding label.

    Output:
        logits (max_label_length, batch_size, action_vocab_size): predicted outputs.
            max_label_length is the maximum length of the labels.
    """

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = nn.Linear(
            decoder.hidden_size,
            decoder.action_vocab_size,
        )
        d = 2 if encoder.bidirectional else 1
        self.decoder_cell_init = nn.Linear(d * encoder.hidden_size, decoder.hidden_size)

    def _init_decoder_state(self, encoder_output, encoder_state, batch_size):
        nlayers = self.decoder.nlayers
        hidden_size = self.decoder.hidden_size
        decoder_hidden = torch.zeros((nlayers, batch_size, hidden_size)).type_as(
            encoder_output
        )

        encoder_hidden, encoder_cell = encoder_state
        encoder_cell = encoder_cell.transpose(0, 1).reshape(batch_size, -1)
        decoder_cell = self.decoder_cell_init(encoder_cell).unsqueeze(0)
        return decoder_hidden, decoder_cell

    def forward(self, input, label, input_length, label_length):
        batch_size = len(input_length)
        sentence_mask = (
            unpack(pack(input, input_length.cpu(), enforce_sorted=False))[0] == 0
        )

        encoder_output, _, encoder_state = self.encoder(input, input_length)

        # accumulate all attentional outputs
        decoder_att_outputs = []

        decoder_state = self._init_decoder_state(
            encoder_output, encoder_state, batch_size
        )
        decoder_att_output = torch.zeros(
            (batch_size, self.decoder.hidden_size),
        ).type_as(input)
        for t in range(max(label_length)):
            decoder_att_output, decoder_state = self.decoder(
                label[t],
                decoder_att_output,
                decoder_state,
                encoder_output,
                sentence_mask,
            )
            decoder_att_outputs.append(decoder_att_output)

        logits = self.predictor(torch.stack(decoder_att_outputs))
        return logits


class EncoderLSTM(nn.Module):
    """Simple LSTM encoder with dropout.

    Inputs:
        input (N1, batch_size, word_vocab_size): input words.
        input_length (batch_size): length of non-padding input.
        state: (optional) previous encoder state; by default, will initialize with zeros.

    Outputs:
        output (max_sentence_length, batch_size, hidden_size): the encoder output.
            max_sentence_length is the length of the longest input.
        output_length (batch_size): length of the encoder output.
        state: the final state of the encoder.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        nlayers,
        dropout_p,
        bidirectional,
    ):
        super(EncoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            nlayers,
            bidirectional=bidirectional,
        )

    def forward(self, input, length, state=None):
        embeds = self.embedding(input)
        dropped = self.dropout(embeds)
        packed = pack(dropped, length.cpu(), enforce_sorted=False)
        output, state = self.lstm(packed, state)
        unpacked_output, length = unpack(output)
        return unpacked_output, length, state


class DecoderLSTM(nn.Module):
    """LSTM decoder with dropout and attention.

    Inputs:
        prev_action (batch_size): the previous action.
        prev_state: the previous decoder state.
        prev_att_output (batch_size, hidden_size): previous attentional decoder output.
        encoder_output (max_sentence_length, batch_size, encoder_hidden_size):
            output of the encoder for calculating attention.
        encoder_output_length (batch_size):
            length of non-padding encoder output.

    Outputs:
        att_output (batch_size, hidden_size): attentional decoder output.
        state: the new decoder state.
    """

    def __init__(
        self,
        action_vocab_size,
        embedding_dim,
        hidden_size,
        nlayers,
        dropout_p,
        context_size,
    ):
        super(DecoderLSTM, self).__init__()
        self.action_vocab_size = action_vocab_size
        self.embedding_size = embedding_dim
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout_p = dropout_p
        # if bidirectional, context_size should be 2 * encoder.hidden_size
        self.context_size = context_size

        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(action_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_size,
            hidden_size,
            nlayers,
        )
        self.merge_context_output = nn.Linear(context_size + hidden_size, hidden_size)

    def _calculate_attention(self, query, key, value, sentence_mask):
        """Perform standand attention calculation.

        Args:
            query (batch_size, d1)
            key (batch_size, d1)
            value (max_sentence_length, batch_size, d2)
            value_mask (max_sentence_length, batch_size):
                boolean tensor indicating paddings in value.

        Returns:
            context (batch_size, d2): a weighed-sum of values
        """
        # (max_sentence_length, batch_size)
        logits = (query * key).sum(-1)
        logits.masked_fill_(sentence_mask, -float("inf"))
        # (max_sentence_length, batch_size)
        probs = F.softmax(logits, dim=-1)
        context = torch.sum(probs.unsqueeze(2) * value, dim=0)
        return context

    def _calculate_att_output(self, context, output):
        """Add context to an output to get an attentional output."""
        att_output = torch.cat([context, output], dim=1)
        att_output = self.dropout(att_output)
        att_output = torch.tanh(self.merge_context_output(att_output))
        return att_output

    def forward(
        self,
        prev_action,
        prev_att_output,
        prev_state,
        encoder_output,
        sentence_mask,
    ):
        # input_embed (batch_size, action_vocab_size)
        embed_prev_action = self.embedding(prev_action)

        # TODO: add parent feeding
        # lstm_input (batch_size, action_vocab_size + hidden_size)
        lstm_input = torch.cat([embed_prev_action, prev_att_output], dim=1)
        output, state = self.lstm(lstm_input.unsqueeze(0), prev_state)
        output = output.squeeze(0)  # (batch_size, hidden_size)

        # prepare the next attentional output
        context = self._calculate_attention(
            query=prev_att_output,
            key=encoder_output,
            value=encoder_output,
            sentence_mask=sentence_mask,
        )

        att_output = self._calculate_att_output(context, output)

        return att_output, state
