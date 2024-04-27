from torch import nn

from gen_names.config import Params

class LSTM(nn.Module):
    def __init__(self, args: Params) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(args.model.vocab_size, args.model.embedding_dim)

        self.rnn = nn.LSTM(
            input_size = args.model.embedding_dim, 
            hidden_size = args.model.hidden_state, 
            num_layers = args.model.num_layers,
            batch_first = True,
            dropout = args.model.dropout
        )

        self.head = nn.Linear(
            in_features = args.model.hidden_state,
            out_features = args.model.vocab_size
        )

    def forward(self, x):
        x = self.embeddings(x)
        x, (_, _) = self.rnn(x)
        x = self.head(x)
        return x