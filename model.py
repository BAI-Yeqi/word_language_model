import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNNModel(nn.Module):
    """Container module with an encoder, a feed-forward module, and a decoder."""

    def __init__(self, seq_len, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        '''
        Arguments:
            seq_len: the input sequence length, which shall be fixed for a Feed-forward Network
        '''
        super(FNNModel, self).__init__()
        self.seq_len = seq_len
        # Reserve a token for padding
        self.pad = ntoken
        ntoken = ntoken + 1
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.fc = nn.Linear(seq_len*ninp, nhid)
        self.tanh = nn.Tanh()
        self.decoder = nn.Linear(nhid, ntoken)

        # Tie weights: weight sharing between encoder and decoder
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        nn.init.uniform_(self.fc.weight, -initrange, initrange)

    def wrap_input(self, input):
        '''
        Preprocess the input to fit the computation graph of FNNModel
        e.g. input = [[1, 3], 
                      [2, 4]]
             wrapped_input = [
                 [[<PAD>, 1], [<PAD>, 3], 
                 [[1, 2]], [3, 4]]
             ]
        Arguments:
            input: torch tensor with shape [seq_len, batch_size]
        Returns:
            wrapped_input: torch tensor with shape [seq_len, batch_size, model_seq_len]
        '''
        wrapped_input = []
        batch_size = input.shape[1]
        seq_len = input.shape[0]
        # Move along the time dimension
        for step_id in range(0, seq_len):
            if step_id == self.seq_len-1:
                # The last time step needs no padding
                wrapped_input.append(input)
                continue
            # Otherwise, get available tokens until this step
            valid_tokens = input[0:step_id+1, :]
            padding = self.pad * torch.ones(
                [self.seq_len-1-step_id, batch_size], dtype=torch.int32
            ).to(valid_tokens.device)
            padded_tokens = torch.cat(
                [padding, valid_tokens], dim=0)
            wrapped_input.append(padded_tokens)
        # [seq_len, seq_len, batch_size]
        wrapped_input = torch.stack(wrapped_input, dim=0)
        # [seq_len, batch_size, seq_len]
        wrapped_input = torch.transpose(wrapped_input, 1, 2)
        return wrapped_input
        
    def forward(self, input):
        #print(input.shape, self.seq_len)
        wrapped_input = self.wrap_input(input)
        #print("wrapped_input", wrapped_input.shape)
        valid_len = wrapped_input.shape[2]==self.seq_len
        assert valid_len, "wrapped seq_len must be same to FNNModel.seq_len"
        emb = self.drop(self.encoder(wrapped_input))
        #print("emb", emb.shape)
        emb = emb.view(emb.shape[0], emb.shape[1], -1)
        #print("emb", emb.shape)
        output = self.fc(emb)
        output = self.tanh(output)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1)

    def get_word_sim(self, w1, w2):
        '''
        Return the cosine similarity between w1 and w2
        '''
        e1 = self.encoder(w1)
        e2 = self.encoder(w2)
        return F.cosine_similarity(e1, e2, dim=0)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        #self.inspect_forward(input, hidden)
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def inspect_forward(self, input, hidden):
        print('input:', input.shape)
        print('input:', input)
        print('hidden[0]:', hidden[0].shape)
        emb = self.drop(self.encoder(input))
        print('emb:', emb.shape)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        print('output:', output.shape)
        print('hidden[0]:', hidden[0].shape)
        decoded = self.decoder(output)
        print('decoded:', decoded.shape)
        decoded = decoded.view(-1, self.ntoken)
        print('decoded:', decoded.shape)
        return F.log_softmax(decoded, dim=1), hidden

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


if __name__ == '__main__':
    fnn_model = FNNModel(10, 1000, 200, 200, 1, 0.5)
    fnn_model.init_weights()
    sim = fnn_model.get_word_sim(torch.tensor(3), torch.tensor(5))
    print(sim)
