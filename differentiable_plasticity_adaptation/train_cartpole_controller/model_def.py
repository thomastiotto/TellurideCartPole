from torch import nn


def define_ff_model_struct(hidden_size, input_size, depth):
    hidden_layers = []
    for _ in range(depth):
        hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        hidden_layers.append(nn.Tanh())

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        *hidden_layers,
        nn.Linear(hidden_size, 1),
    )

    return model


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


def define_rec_model_struct(hidden_size, input_size, depth):
    model = nn.Sequential(
        # nn.Linear(input_size, hidden_size),
        # nn.Tanh(),
        nn.GRU(input_size, hidden_size, depth),
        SelectItem(0),
        nn.Linear(hidden_size, 1),
    )

    return model


class model(nn.Module):
    def __init__(self, ninp, num_layers, nhid):
        super().__init__()

        self.gru_nets = nn.GRU(input_size=ninp, hidden_size=nhid, num_layers=num_layers,
                               batch_first=True, dropout=0.2, bidirectional=False)
        self.FC = nn.Linear(nhid, 1)
        self.tanh = nn.Tanh()

    def forward(self, X, hidden=None):
        if hidden is None:
            out, hidden_new = self.gru_nets(X)
        else:
            out, hidden_new = self.gru_nets(X, hidden)

        # out = out[:, -1]

        out = self.tanh(out)

        out = self.FC(out)

        return out, hidden_new
