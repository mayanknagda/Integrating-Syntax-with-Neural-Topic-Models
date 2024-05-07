import os
import sys
import time
import torch
import pickle
import torch.nn as nn
import numpy as np

sys.path.append('...')
from src.models.cn import LM
import torch.nn.functional as F


def train_cn(train_dl, val_dl, test_dl, context_type, context_size, vocab_embeddings, path, w2idx, history):
    history.append((time.asctime(), 'start training context network'))
    if context_type == 'symmetric':
        input_dim = context_size * vocab_embeddings.shape[1] * 2
    elif context_type == 'asymmetric':
        input_dim = context_size * vocab_embeddings.shape[1]
    else:
        NotImplementedError(f'context_type {context_type} not implemented')

    vocab_size = vocab_embeddings.shape[0]

    net = LM(input_dim=input_dim, output_dim=vocab_size, vocab_embeddings=vocab_embeddings)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    valid_loss_min = np.Inf
    for epoch in range(1):
        net.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()
            y_hat = net(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        history.append((time.asctime(), f"Epoch: {epoch + 1}...Train loss: {train_loss:.6f}"))
        net.eval()
        val_loss = 0
        for i, (x, y) in enumerate(val_dl):
            y_hat = net(x)
            loss = criterion(y_hat, y)
            val_loss += loss.item()
        val_loss /= len(val_dl)
        history.append((time.asctime(), f"Epoch: {epoch + 1}...Validation loss: {val_loss:.6f}"))
        if val_loss <= valid_loss_min:
            history.append((time.asctime(), f"Validation loss decreased ({valid_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."))
            torch.save(net.state_dict(), os.path.join(path, 'cn.pt'))
            valid_loss_min = val_loss
    net.load_state_dict(torch.load(os.path.join(path, 'cn.pt')))
    net.eval()
    threshold_p = [0.1, 0.3, 0.5]
    top_k = [1, 3, 5]
    for p in threshold_p:
        content_vector, syntax_vector, content_words, syntax_words = decision_module_p(p=p, model=net, train_dl=train_dl, w2idx=w2idx, device='cpu')
        with open(os.path.join(path, f'syntax_words_threshold_{p}.txt'), 'w') as f:
            for word in syntax_words:
                f.write(word + "\n")
        # content_words to .txt file
        with open(os.path.join(path, f'content_words_threshold_{p}.txt'), 'w') as f:
            for word in content_words:
                f.write(word + "\n")
        pickle.dump(content_vector, open(os.path.join(path, f'content_vector_threshold_{p}.pkl'), 'wb'))
        pickle.dump(syntax_vector, open(os.path.join(path, f'syntax_vector_threshold_{p}.pkl'), 'wb'))
        pickle.dump(content_words, open(os.path.join(path, f'content_words_threshold_{p}.pkl'), 'wb'))
        pickle.dump(syntax_words, open(os.path.join(path, f'syntax_words_threshold_{p}.pkl'), 'wb'))

    for k in top_k:
        content_vector, syntax_vector, content_words, syntax_words = decision_module_top(n=k, model=net, train_dl=train_dl, w2idx=w2idx, device='cpu')
        with open(os.path.join(path, f'syntax_words_top_{k}.txt'), 'w') as f:
            for word in syntax_words:
                f.write(word + "\n")
        # content_words to .txt file
        with open(os.path.join(path, f'content_words_top_{k}.txt'), 'w') as f:
            for word in content_words:
                f.write(word + "\n")
        pickle.dump(content_vector, open(os.path.join(path, f'content_vector_top_{k}.pkl'), 'wb'))
        pickle.dump(syntax_vector, open(os.path.join(path, f'syntax_vector_top_{k}.pkl'), 'wb'))
        pickle.dump(content_words, open(os.path.join(path, f'content_words_top_{k}.pkl'), 'wb'))
        pickle.dump(syntax_words, open(os.path.join(path, f'syntax_words_top_{k}.pkl'), 'wb'))

    return history


def decision_module_p(p, model, train_dl, w2idx, device):
    dec_vector = torch.zeros(len(w2idx))
    idx2word = {v: k for k, v in w2idx.items()}
    pad_idx = w2idx['[PAD]']
    with torch.no_grad():
        for idx, (x, y) in enumerate(train_dl):
            x = x.to(device)
            # x is of shape (batch_size, context_size)
            y = y.to(device)
            # y is of shape (batch_size)
            y_pred = model(x)
            y_pred = F.softmax(y_pred, dim=-1)
            # y_pred is of shape (batch_size, vocab_size)
            y_pred = y_pred.cpu().numpy()
            y_pred = y_pred > p
            y = y.cpu().numpy()
            for i in range(x.shape[0]):
                if y[i] != pad_idx:
                    if y_pred[i, y[i]]:
                        # syntax
                        dec_vector[y[i]] += 1
                    else:
                        # content
                        dec_vector[y[i]] -= 1
    content_vector = dec_vector < 0
    syntax_vector = dec_vector > 0
    syntax_words = [idx2word[i] for i in range(len(w2idx)) if syntax_vector[i]]
    content_words = [idx2word[i] for i in range(len(w2idx)) if content_vector[i]]
    return content_vector, syntax_vector, content_words, syntax_words


def decision_module_top(n, model, train_dl, w2idx, device):
    pad_idx = w2idx['[PAD]']
    dec_vector = torch.zeros(len(w2idx))
    idx2word = {v: k for k, v in w2idx.items()}
    with torch.no_grad():
        for idx, (x, y) in enumerate(train_dl):
            x = x.to(device)
            # x is of shape (batch_size, context_size)
            y = y.to(device)
            # y is of shape (batch_size)
            y_pred = model(x)
            # y_pred is of shape (batch_size, vocab_size)
            top_3 = torch.topk(y_pred, n, dim=-1).indices
            # top_3 is of shape (batch_size, 3)
            top_3 = top_3.cpu().numpy()
            y = y.cpu().numpy()
            for i in range(x.shape[0]):
                if y[i] != pad_idx:
                    if y[i] in top_3[i]:
                        # syntax
                        dec_vector[y[i]] += 1
                    else:
                        # content
                        dec_vector[y[i]] -= 1
    content_vector = dec_vector < 0
    syntax_vector = dec_vector > 0
    syntax_words = [idx2word[i] for i in range(len(w2idx)) if syntax_vector[i]]
    content_words = [idx2word[i] for i in range(len(w2idx)) if content_vector[i]]
    return content_vector, syntax_vector, content_words, syntax_words


if __name__ == '__main__':
    train_cn(None, None, None, 'word', 4, torch.randn(20, 100))
