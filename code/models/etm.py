import torch
import torch.nn as nn
from torch.distributions import Dirichlet, LogNormal
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_size: int):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=vocab_size, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=500, out_features=num_topics),
            nn.BatchNorm1d(num_features=num_topics, affine=False),
            nn.Softplus()
        )

    def forward(self, x):
        alpha = self.encoder(x)
        alpha = torch.max(torch.tensor(0.00001, device=x.device), alpha)
        return alpha

class EncoderLN(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_size: int):
        super(EncoderLN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=vocab_size, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=500, out_features=2 * num_topics),
            nn.BatchNorm1d(num_features=2 * num_topics, affine=False),
            nn.Softplus()
        )

    def forward(self, x):
        alpha = self.encoder(x)
        return alpha


class Decoder(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_embeddings: int):
        super(Decoder, self).__init__()
        vocab_size, embed_dim = vocab_embeddings.shape
        # embedding decoder
        self.topic_embeddings = nn.Linear(num_topics, embed_dim, bias=False)
        self.word_embeddings = nn.Linear(embed_dim, vocab_size, bias=False)
        # initialize linear layer with pre-trained embeddings
        self.word_embeddings.weight.data.copy_(vocab_embeddings)
        # self.word_embeddings.weight.requires_grad = False
        self.decoder_norm = nn.Sequential(
            nn.BatchNorm1d(vocab_size, affine=False),
            nn.Softmax(dim=1),
        )

    def forward(self, z):
        topic_embeddings = self.topic_embeddings(z)  # (batch_size, 300)
        word_embeddings = self.word_embeddings.weight  # (vocab_size, 300)
        # dot product
        recon = torch.matmul(topic_embeddings, word_embeddings.T)  # (batch_size, vocab_size)
        recon = self.decoder_norm(recon)  # (batch_size, vocab_size)
        return recon


class ETM(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_embeddings: int):
        super(ETM, self).__init__()
        vocab_size, embed_dim = vocab_embeddings.shape
        self.num_topics = num_topics
        self.encoder = EncoderLN(num_topics, vocab_size)
        self.decoder = Decoder(num_topics, vocab_embeddings)

    def forward(self, x):
        alpha = self.encoder(x)
        mu = alpha[:, :self.num_topics]
        sigma = alpha[:, self.num_topics:]
        sigma = torch.max(torch.tensor(0.00001, device=x.device), sigma)
        dist = LogNormal(mu, sigma)
        if self.training:
            dist_sample = dist.rsample()
        else:
            dist_sample = dist.mean
        # decoders
        dist_sample = F.softmax(dist_sample, dim=-1)
        x_recon = self.decoder(dist_sample)
        return x_recon, dist


class ETMD(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_embeddings: int):
        super(ETMD, self).__init__()
        vocab_size, embed_dim = vocab_embeddings.shape
        self.encoder = Encoder(num_topics, vocab_size)
        self.decoder = Decoder(num_topics, vocab_embeddings)

    def forward(self, x):
        alpha = self.encoder(x)
        dist = Dirichlet(alpha)
        if self.training:
            dist_sample = dist.rsample()
        else:
            dist_sample = dist.mean
        # decoders
        dist_sample = F.softmax(dist_sample, dim=-1)
        x_recon = self.decoder(dist_sample)
        return x_recon, dist


class SyConETM(nn.Module):
    def __init__(self,
                 syn_topics: int,
                 sem_topics: int,
                 vocab_embeddings: int):
        super(SyConETM, self).__init__()
        vocab_size, embed_dim = vocab_embeddings.shape
        self.sem_topics = sem_topics
        self.syn_topics = syn_topics
        self.encoder = Encoder(syn_topics + sem_topics, vocab_size)
        self.decoder_syn = Decoder(syn_topics, vocab_embeddings)
        self.decoder_sem = Decoder(sem_topics, vocab_embeddings)

    def forward(self, x):
        alpha = self.encoder(x)
        alpha_sem = alpha[:, :self.sem_topics]
        alpha_syn = alpha[:, self.sem_topics:]
        dist_sem = Dirichlet(alpha_sem)
        dist_syn = Dirichlet(alpha_syn)
        if self.training:
            z_sem = dist_sem.rsample()
            z_syn = dist_syn.rsample()
        else:
            z_sem = dist_sem.mean
            z_syn = dist_syn.mean
        x_recon_sem = self.decoder_sem(z_sem)
        x_recon_syn = self.decoder_syn(z_syn)
        prior_sem = torch.ones_like(alpha_sem) * 0.02
        prior_syn = torch.ones_like(alpha_syn) * 0.02
        kl_sem = torch.distributions.kl_divergence(dist_sem, Dirichlet(prior_sem)).mean()
        kl_syn = torch.distributions.kl_divergence(dist_syn, Dirichlet(prior_syn)).mean()
        return x_recon_sem, x_recon_syn, kl_sem, kl_syn


if __name__ == '__main__':
    vocab_embeddings = torch.rand(size=(100, 300))
    net = ETMD(num_topics=20, vocab_embeddings=vocab_embeddings)
    t = torch.rand(size=(32, 100))
    print(net(t)[0].shape)
    net = SyConETM(syn_topics=10, sem_topics=20, vocab_embeddings=vocab_embeddings)
    t = torch.rand(size=(32, 100))
    beta = net.decoder_syn.topic_embeddings.weight.T @ net.decoder_syn.word_embeddings.weight.T
    print(beta.shape)
    beta = net.decoder_sem.topic_embeddings.weight.T @ net.decoder_sem.word_embeddings.weight.T
    print(beta.shape)
    print(net(t)[0].shape)
    print(net(t)[1].shape)
