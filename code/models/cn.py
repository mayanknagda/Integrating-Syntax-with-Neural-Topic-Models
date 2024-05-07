import torch
import torch.nn as nn


class LM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 vocab_embeddings: torch.Tensor) -> None:
        super().__init__()
        self.vocab_embeddings = nn.Embedding.from_pretrained(vocab_embeddings, freeze=False)
        self.language_model = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=input_dim, out_features=2048),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(num_features=2048),
            nn.Linear(in_features=2048, out_features=output_dim)
        )

    def forward(self, x):
        # x is a 3D tensor of shape (batch_size, context_size)
        x = self.vocab_embeddings(x)
        # x is a 3D tensor of shape (batch_size, context_size, embedding_dim)
        # now we need to reshape it to (batch_size, context_size*embedding_dim)
        x = x.reshape(x.shape[0], -1)
        # x is a 2D tensor of shape (batch_size, context_size*embedding_dim)
        return self.language_model(x)


if __name__ == '__main__':
    net = LM(input_dim=400, output_dim=20, vocab_embeddings=torch.randn(20, 100))
    t = torch.ones(size=(32, 4)).long()
    print(net(t).shape)