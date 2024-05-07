import os
import sys
import time
import torch
import pickle
import torch.nn as nn
import numpy as np

sys.path.append('...')
from src.models.dvae import SyConDVAE
from src.models.etm import SyConETM
import torch.nn.functional as F


def train_syconntm(model_name, train_dl, val_dl, test_dl, syn_topics, sem_topics, syn_vec, sem_vec, vocab_embeddings, path, history):
    history.append((time.asctime(), 'start training syconntm model: {}'.format(model_name)))
    vocab_size, embed_dim = vocab_embeddings.shape
    if 'dvae' in model_name:
        model = SyConDVAE(syn_topics=syn_topics, sem_topics=sem_topics, vocab_size=vocab_size)
    elif 'etm' in model_name:
        model = SyConETM(syn_topics=syn_topics, sem_topics=sem_topics, vocab_embeddings=vocab_embeddings)
    else:
        NotImplementedError(f'{model_name} not implemented')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    valid_loss_min = np.Inf
    history.append((time.asctime(), 'start training'))
    for epoch in range(1):
        train_loss = 0
        model.train()
        for i, (x) in enumerate(train_dl):
            optimizer.zero_grad()
            x = x[0].float().to(device)
            x_recon_sem, x_recon_syn, kl_sem, kl_syn = model(x)
            x_sem = x * sem_vec
            x_syn = x * syn_vec
            recon_loss = -torch.sum(x_recon_sem * x_sem, dim=1).mean() - torch.sum(x_recon_syn * x_syn, dim=1).mean()
            kl_loss = kl_sem + kl_syn
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        history.append((time.asctime(), "Epoch: {}.. Train Loss: .. ".format(epoch + 1)))
        model.eval()
        val_loss = 0
        for i, (x) in enumerate(val_dl):
            x = x[0].float().to(device)
            x_recon_sem, x_recon_syn, kl_sem, kl_syn = model(x)
            x_sem = x * sem_vec
            x_syn = x * syn_vec
            recon_loss = -torch.sum(x_recon_sem * x_sem, dim=1).mean() - torch.sum(x_recon_syn * x_syn, dim=1).mean()
            kl_loss = kl_sem + kl_syn
            loss = recon_loss + kl_loss
            val_loss += loss.item()
        val_loss /= len(val_dl)
        history.append((time.asctime(), "Epoch: {}.. Val Loss: .. ".format(epoch + 1, val_loss)))
        if val_loss <= valid_loss_min:
            history.append((time.asctime(), "Epoch: {}.. Val Loss: .. ".format(epoch + 1, val_loss)))
            torch.save(model.state_dict(), os.path.join(path, f'{model_name}.pt'))
            valid_loss_min = val_loss
    model.load_state_dict(torch.load(os.path.join(path, f'{model_name}.pt')))
    model.eval()
    # get beta here
    if 'dvae' in model_name:
        beta_syn = model.decoder_syn.decoder.weight.cpu().detach().numpy().T
        beta_sem = model.decoder_sem.decoder.weight.cpu().detach().numpy().T
    elif 'etm' in model_name:
        te = model.decoder_syn.topic_embeddings.weight.cpu().detach().numpy().T
        we = model.decoder_syn.word_embeddings.weight.cpu().detach().numpy().T
        beta_syn = te @ we
        te = model.decoder_sem.topic_embeddings.weight.cpu().detach().numpy().T
        we = model.decoder_sem.word_embeddings.weight.cpu().detach().numpy().T
        beta_sem = te @ we
    # save beta
    with open(os.path.join(path, f'{model_name}_beta.pkl'), 'wb') as f:
        pickle.dump((beta_syn, beta_sem), f)
    return history
