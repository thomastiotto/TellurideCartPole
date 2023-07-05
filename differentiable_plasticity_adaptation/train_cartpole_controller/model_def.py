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
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.GRU(hidden_size, hidden_size, depth),
        SelectItem(1),
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
    )

    return model
