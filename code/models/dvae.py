import torch
import torch.nn as nn
from torch.distributions import Dirichlet


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


class Decoder(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_size: int):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(in_features=num_topics, out_features=vocab_size)
        self.decoder_norm = nn.Sequential(
            nn.BatchNorm1d(vocab_size, affine=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, z):
        x_recon = self.decoder_norm(self.decoder(z))
        return x_recon


class DVAE(nn.Module):
    def __init__(self,
                 num_topics: int,
                 vocab_size: int):
        super(DVAE, self).__init__()
        self.encoder = Encoder(num_topics, vocab_size)
        self.decoder = Decoder(num_topics, vocab_size)

    def forward(self, x):
        alpha = self.encoder(x)
        dist = Dirichlet(alpha)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        x_recon = self.decoder(z)
        return x_recon, dist


class SyConDVAE(nn.Module):
    def __init__(self,
                 syn_topics: int,
                 sem_topics: int,
                 vocab_size: int):
        super(SyConDVAE, self).__init__()
        self.sem_topics = sem_topics
        self.syn_topics = syn_topics
        self.encoder = Encoder(syn_topics + sem_topics, vocab_size)
        self.decoder_syn = Decoder(syn_topics, vocab_size)
        self.decoder_sem = Decoder(sem_topics, vocab_size)

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
    # net = DVAE(num_topics=20, vocab_size=100)
    # t = torch.rand(size=(32, 100))
    # beta = net.decoder.decoder.weight.T
    # print(beta.shape)
    # print(net(t)[0].shape)

    net = SyConDVAE(syn_topics=10, sem_topics=20, vocab_size=100)
    t = torch.rand(size=(32, 100))
    beta = net.decoder_syn.decoder.weight.T
    print(beta.shape)
    beta = net.decoder_sem.decoder.weight.T
    print(beta.shape)
    print(net(t)[0].shape)
    print(net(t)[1].shape)

