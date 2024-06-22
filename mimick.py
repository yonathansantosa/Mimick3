# %% [markdown]
# # Initialization

# %%
import smart_open
import gensim.downloader
import numpy as np
import torch
import torch.nn as nn
import os
import wandb
import requests
import pprint
os.environ["WANDB_MODE"]="offline"
# %%
wandb.login(key='f8dba7836d4d8c528b40ebd197a992eb44f9c29f')

# %% [markdown]
# # Embeddings

# %%
#@title Imports
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from polyglot.mapping import Embedding
from torchtext.vocab import Vectors, GloVe
from polyglot.downloader import downloader

# %%
#@title PCA
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        X = X.to(device)
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.mean_ = X.mean(0, keepdim=True)
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh.to(device)
        U = U.to(device)
        U, Vt = svd_flip(U, Vt)
        self.components_ = Vt[:d]
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        # print(Y.get_device())
        # print(self.components_.get_device())
        return torch.matmul(Y, self.components_) + self.mean_
    
# %%
#@title Word_embedding
class Word_embedding:
    def __init__(self, emb_dim=300, w2v_size=50000, lang='en', embedding='polyglot', pca=None):
        '''
        Initializing word embedding
        Parameter:
        emb_dim = (int) embedding dimension for word embedding
        '''
        self.embedding_vectors = None
        if pca: 
            self.pca = PCA(pca)
            self.pca.to(device)
        if embedding == 'glove':
            # *GloVE
            glove = GloVe('6B', dim=emb_dim)
            self.embedding_vectors = glove.vectors
            self.stoi = glove.stoi
            self.itos = glove.itos
        elif embedding == 'word2vec':
            # *word2vec
            # word2vec = Vectors('GoogleNews-vectors-negative300.bin.gz.txt')
            w2v_vectors = gensim.downloader.load('word2vec-google-news-300')
            self.embedding_vectors = torch.from_numpy(w2v_vectors.vectors[:w2v_size])
            self.stoi = w2v_vectors.key_to_index
            self.itos = w2v_vectors.index_to_key[:w2v_size]
        elif embedding == 'polyglot':
            # *Polyglot
            if not os.path.exists('polyglot/embeddings2/en/embeddings_pkl.tar.bz2'):
                downloader.download("embeddings2.en", download_dir='polyglot')
            polyglot_emb = Embedding.load('polyglot/embeddings2/%s/embeddings_pkl.tar.bz2' % lang)
            self.embedding_vectors = torch.from_numpy(polyglot_emb.vectors)
            self.stoi = polyglot_emb.vocabulary.word_id
            self.itos = [polyglot_emb.vocabulary.id_word[i] for i in range(len(polyglot_emb.vocabulary.id_word))]
        elif embedding == 'dict2vec':
            if not os.path.exists('dict2vec-100d.vec'):
                response = requests.get('https://raw.githubusercontent.com/yonathansantosa/Mimick/master/dict2vec-100d.vec')
                with open("dict2vec-100d.vec", mode="wb") as file:
                    file.write(response.content)
            word2vec = Vectors('dict2vec-100d.vec')
            self.embedding_vectors = word2vec.vectors
            self.stoi = word2vec.stoi
            self.itos = word2vec.itos
        # print(self.embedding_vectors.shape)

        if pca:
            self.embedding_vectors_pca = self.pca.fit_transform(self.embedding_vectors.to(device))
            # print(self.embedding_vectors_pca.shape)
            self.word_embedding = nn.Embedding.from_pretrained(self.embedding_vectors_pca, freeze=True, sparse=True)
            self.emb_dim = pca
        else:
            self.word_embedding = nn.Embedding.from_pretrained(self.embedding_vectors.to(device), freeze=True, sparse=True)
            self.emb_dim = self.embedding_vectors.size(1)
        

    def __getitem__(self, index):
        return (torch.tensor([index], dtype=torch.long).to(device), self.word_embedding(torch.tensor([index], device=device)).squeeze())

    def __len__(self):
        return len(self.itos)

    def update_weight(self, weight):
        new_emb = Vectors(weight)
        self.word_embedding = nn.Embedding.from_pretrained(self.embedding_vectors, freeze=True, sparse=True)
        if pca:
            self.emb_dim = self.embedding_vectors_pca.size(1)
        else:
            self.emb_dim = self.embedding_vectors.size(1)
        self.stoi = new_emb.stoi
        self.itos = new_emb.itos

    def word2idx(self, c):
        return self.stoi[c]

    def idx2word(self, idx):
        return self.itos[int(idx)]

    def idxs2sentence(self, idxs):
        return ' '.join([self.itos[int(i)] for i in idxs])

    def sentence2idxs(self, sentence):
        word = sentence.split()
        return [self.stoi[w] for w in word]

    def idxs2words(self, idxs):
        '''
        Return tensor of indexes as a sentence

        Input:
        idxs = (torch.LongTensor) 1D tensor contains indexes
        '''
        idxs = idxs.squeeze()
        sentence = [self.itos[int(idx)] for idx in idxs]
        return sentence

    def get_word_vectors(self):
        return self.word_embedding

# %%
#@title Word_embedding_test
class Word_embedding_test:
    def __init__(self, emb_dim=300):
        '''
        Initializing word embedding
        Parameter:
        emb_dim = (int) embedding dimension for word embedding
        '''
        self.embedding = './.vector_cache/GoogleNews-vectors-negative300.bin.gz-'
        self.stoi = "./.vector_cache/stoi.txt"
        self.emb_dim = 0

        with open('%s%d.txt' % (self.embedding, 0), encoding='utf-8') as fp:
            for line in fp:
                entry = line.split(' ')
                self.emb_dim = len(entry) - 1

    def __getitem__(self, index):
        file_id = index // 100000
        file_line = index % 100000
        vector = None
        with open('%s%d.txt' % (self.embedding, file_id), encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                if i == file_line:
                    entry = line.split(' ')
                    vector = np.array(entry[1:], dtype=np.float32)
                elif i > file_line:
                    break
        return (torch.tensor([file_id, file_line], dtype=torch.long), torch.tensor(vector))

    def __len__(self):
        with open(self.stoi, encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def idx2word(self, file_idx, file_line):
        with open('%s%d.txt' % (self.embedding, file_idx), encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                if i == file_line:
                    entry = line.split(' ')
                    return(entry[0])
                elif i > file_line:
                    break

    # def idxs2sentence(self, idxs):
    #     return ' '.join([self.itos[int(i)] for i in idxs])

    # def sentence2idxs(self, sentence):
    #     word = sentence.split()
    #     return [self.stoi[w] for w in word]

    def idxs2words(self, idxs):
        '''
        Return tensor of indexes as a sentence

        Input:
        idxs = (torch.LongTensor) 1D tensor contains indexes
        '''
        return [self.idx2word(idx, line) for idx, line in idxs]

    # def get_word_vectors(self):
    #     return self.word_embedding

# %%
#@title Char Embedding
class Char_embedding:
    def __init__(self, char_emb_dim=300, char_max_len=15, random=False, asc=False, device='cuda', freeze=False):
        super(Char_embedding, self).__init__()
        '''
        Initializing character embedding
        Parameter:
        emb_dim = (int) embedding dimension for character embedding
        ascii = mutually exclusive with random
        '''
        self.char_max_len = char_max_len
        self.asc = asc
        if random and not self.asc:
            torch.manual_seed(5)
            if not os.path.exists('glove.840B.300d-char.txt'):
                response = requests.get('https://raw.githubusercontent.com/yonathansantosa/Mimick/master/glove.840B.300d-char.txt')
                with open("glove.840B.300d-char.txt", mode="wb") as file:
                    file.write(response.content)
            table = np.transpose(np.loadtxt('glove.840B.300d-char.txt', dtype=str, delimiter=' ', comments='##'))
            self.weight_char = np.transpose(table[1:].astype(float))
            self.char = np.transpose(table[0])
            self.embed = nn.Embedding(len(self.char), char_emb_dim).to(device)
            None
        elif self.asc:
            if not os.path.exists('ascii.embedding.txt'):
                response = requests.get('https://raw.githubusercontent.com/yonathansantosa/Mimick/master/ascii.embedding.txt')
                with open("ascii.embedding.txt", mode="wb") as file:
                    file.write(response.content)
            table = np.transpose(np.loadtxt('ascii.embedding.txt', dtype=str, delimiter=' ', comments='##'))
            self.char = np.transpose(table[0])
            self.weight_char = np.transpose(table[1:].astype(float))

            self.weight_char = torch.from_numpy(self.weight_char).to(device)

            self.embed = nn.Embedding.from_pretrained(self.weight_char, freeze=freeze)
        else:
            if not os.path.exists('glove.840B.300d-char.txt'):
                response = requests.get('https://raw.githubusercontent.com/yonathansantosa/Mimick/master/glove.840B.300d-char.txt')
                with open("glove.840B.300d-char.txt", mode="wb") as file:
                    file.write(response.content)
            table = np.transpose(np.loadtxt('glove.840B.300d-char.txt', dtype=str, delimiter=' ', comments='##'))
            self.char = np.transpose(table[0])
            self.weight_char = np.transpose(table[1:].astype(float))
            self.weight_char = self.weight_char[:,:char_emb_dim]

            self.weight_char = torch.from_numpy(self.weight_char).to(device)

            self.embed = nn.Embedding.from_pretrained(self.weight_char, freeze=freeze)

        self.embed.padding_idx = 1
        self.char2idx = {}
        self.idx2char = {}
        self.char_emb_dim = char_emb_dim
        for i, c in enumerate(self.char):
            self.char2idx[c] = int(i)
            self.idx2char[i] = c

    def char_split(self, sentence, dropout=0.):
        '''
        Splitting character of a sentences then converting it
        into list of index

        Parameter:

        sentence = list of words
        '''
        char_data = []
        numbers = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
        # split_sentence = sentence.split()
        # split_sentence = sentence.split()

        for word in sentence:
            if word == '<pad>':
                char_data += [[self.char2idx['<pad>']] * self.char_max_len]
            else:
                c = list(word)
                c = ['<sow>'] + c
                if len(c) > self.char_max_len:
                    # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.char_max_len]]
                    c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.char_max_len]]
                elif len(c) <= self.char_max_len:
                    # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
                    c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
                    if len(c_idx) < self.char_max_len: c_idx.append(self.char2idx['<eow>'])
                    for i in range(self.char_max_len-len(c)-1):
                        c_idx.append(self.char2idx['<pad>'])
                char_data += [c_idx]

        char_data = torch.Tensor(char_data).long()
        char_data = F.dropout(char_data, dropout)
        return char_data

    def char_sents_split(self, sentences, dropout=0.):
        '''
        Splitting character of a sentences then converting it
        into list of index

        Parameter:

        sentence = list of words
        '''
        numbers = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
        # split_sentence = sentence.split()
        # split_sentence = sentence.split()

        sents_data = []
        for sentence in sentences:
            char_data = []
            for word in sentence:
                if word == '<pad>':
                    char_data += [[self.char2idx['<pad>']] * self.char_max_len]
                else:
                    c = list(word)
                    c = ['<sow>'] + c
                    if len(c) > self.char_max_len:
                        # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.char_max_len]]
                        c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.char_max_len]]
                    elif len(c) <= self.char_max_len:
                        # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
                        c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
                        if len(c_idx) < self.char_max_len: c_idx.append(self.char2idx['<eow>'])
                        for i in range(self.char_max_len-len(c)-1):
                            c_idx.append(self.char2idx['<pad>'])
                    char_data += [c_idx]

            char_data = torch.Tensor(char_data).long()
            char_data = F.dropout(char_data, dropout)
            sents_data += [char_data]

        return torch.cat(sents_data)

    def char2ix(self, c):
        return self.char2idx[c]

    def ix2char(self, idx):
        return self.idx2char[idx]

    def idxs2word(self, idxs):
        return "".join([self.idx2char[idx] for idx in idxs])

    def word2idxs(self, word):
        char_data = []
        if word != '<pad>':
            chars = list(word)
            chars = ['<sow>'] + chars
            if len(chars) > self.char_max_len:
                # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.char_max_len]]
                c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in chars[:self.char_max_len]]
            elif len(chars) <= self.char_max_len:
                # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
                c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in chars]
                if len(c_idx) < self.char_max_len: c_idx.append(self.char2idx['<eow>'])
                for i in range(self.char_max_len-len(chars)-1):
                    c_idx.append(self.char2idx['<pad>'])
        else:
            c_idx = [self.char2idx['<pad>']] * self.char_max_len

        char_data += c_idx

        return torch.LongTensor(char_data)

    def clean_idxs2word(self, idxs):
        idxs = [i for i in idxs if i != 0 and i != 1 and i != 2 and i != 3]
        return "".join([self.idx2char[idx] for idx in idxs])

    def get_char_vectors(self, words):
        sentence = []
        for idxs in words:
            sentence += [self.char_embedding(idxs)]

        # return torch.unsqueeze(torch.stack(sentence), 1).permute(1, 0, 2)
        return torch.stack(sentence).permute(1, 0, 2)

# %% [markdown]
# # Model

# %%
#@title Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

# %%
#@title Models
# *Device configuration

class mimick(nn.Module):
    def __init__(self, embedding, char_emb_dim,emb_dim, hidden_size):
        super(mimick, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(char_emb_dim, self.hidden_size, 1, bidirectional=True, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
        )

    def forward(self, inputs):
        x = self.embedding(inputs).float()
        _, (hidden_state, _) = self.lstm(x)
        out_cat = (hidden_state[0, :, :] + hidden_state[1, :, :])
        out = self.mlp(out_cat)

        return out

class mimick_cnn(nn.Module):
    def __init__(self, embedding,  char_max_len=15, char_emb_dim=300, emb_dim=300, num_feature=100, random=False, asc=False):
        super(mimick_cnn, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.conv2 = nn.Conv2d(1, num_feature, (2, char_emb_dim), bias=False)
        self.conv3 = nn.Conv2d(1, num_feature, (3, char_emb_dim), bias=False)
        self.conv4 = nn.Conv2d(1, num_feature, (4, char_emb_dim), bias=False)
        self.conv5 = nn.Conv2d(1, num_feature, (5, char_emb_dim), bias=False)
        self.conv6 = nn.Conv2d(1, num_feature, (6, char_emb_dim), bias=False)
        self.conv7 = nn.Conv2d(1, num_feature, (7, char_emb_dim), bias=False)
        self.inputs = None

        self.mlp1 = nn.Sequential(
            nn.Linear(num_feature*6, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.t = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        self.inputs = inputs
        x = self.embedding(self.inputs).float()
        x2 = self.conv2(x).relu().squeeze(-1)
        x3 = self.conv3(x).relu().squeeze(-1)
        x4 = self.conv4(x).relu().squeeze(-1)
        x5 = self.conv5(x).relu().squeeze(-1)
        x6 = self.conv6(x).relu().squeeze(-1)
        x7 = self.conv7(x).relu().squeeze(-1)


        x2_max = F.max_pool1d(x2, x2.shape[2]).squeeze(-1)
        x3_max = F.max_pool1d(x3, x3.shape[2]).squeeze(-1)
        x4_max = F.max_pool1d(x4, x4.shape[2]).squeeze(-1)
        x5_max = F.max_pool1d(x5, x5.shape[2]).squeeze(-1)
        x6_max = F.max_pool1d(x6, x6.shape[2]).squeeze(-1)
        x7_max = F.max_pool1d(x7, x7.shape[2]).squeeze(-1)


        maxpoolcat = torch.cat([x2_max, x3_max, x4_max, x5_max, x6_max, x7_max], dim=1)

        out_cnn = self.mlp1(maxpoolcat)

        out = self.t(out_cnn) * self.mlp2(out_cnn) + (1 - self.t(out_cnn)) * out_cnn

        return out

class mimick_cnn2(nn.Module):
    def __init__(self, embedding,  char_max_len=15, char_emb_dim=300, emb_dim=300, num_feature=100, random=False, asc=False):
        super(mimick_cnn2, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.conv1 = nn.Conv2d(1, num_feature, (2, char_emb_dim))
        self.conv2 = nn.Conv1d(num_feature, num_feature, 2)
        self.conv3 = nn.Conv1d(num_feature, emb_dim, 2)
        self.conv4 = nn.Conv1d(emb_dim, emb_dim, 2)


        self.mlp1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.t = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.embedding(inputs).float()
        x2_conv1 = self.conv1(x).relu().squeeze(-1)
        x2_max1 = F.max_pool1d(x2_conv1, 2).squeeze(-1)
        x2_conv2 = self.conv2(x2_max1).relu()
        x2_max2 = F.max_pool1d(x2_conv2, 2)
        x2_conv3 = self.conv3(x2_max2).relu()
        x2_max3 = F.max_pool1d(x2_conv3, x2_conv3.shape[2]).squeeze(-1)

        # maxpoolcat = torch.cat([x2_max, x3_max, x4_max, x5_max, x6_max, x7_max], dim=2).view(inputs.size(0), -1)

        out_cnn = self.mlp1(x2_max3)

        out = self.t(out_cnn) * self.mlp2(out_cnn) + (1 - self.t(out_cnn)) * out_cnn

        return out

class mimick_cnn3(nn.Module):
    def __init__(self, embedding, char_max_len=15, char_emb_dim=300, emb_dim=300, num_feature=100, mtp=1, random=False, asc=False):
        super(mimick_cnn3, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.conv2 = nn.Conv2d(1, num_feature, (2, embedding.embedding_dim), bias=False)
        self.conv3 = nn.Conv2d(1, num_feature, (3, embedding.embedding_dim), bias=False)
        self.conv4 = nn.Conv2d(1, num_feature, (4, embedding.embedding_dim), bias=False)
        self.conv5 = nn.Conv2d(1, num_feature, (5, embedding.embedding_dim), bias=False)
        self.conv6 = nn.Conv2d(1, num_feature, (6, embedding.embedding_dim), bias=False)
        self.conv7 = nn.Conv2d(1, num_feature, (7, embedding.embedding_dim), bias=False)

        self.featloc = nn.Sequential(
            nn.Linear(num_feature*99, emb_dim),
            nn.Sigmoid()
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-mtp*3, max_val=mtp*3),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-mtp*3, max_val=mtp*3),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.t = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.embedding(inputs).float()
        x2 = self.conv2(x).sigmoid().squeeze(-1)
        x3 = self.conv3(x).sigmoid().squeeze(-1)
        x4 = self.conv4(x).sigmoid().squeeze(-1)
        x5 = self.conv5(x).sigmoid().squeeze(-1)
        x6 = self.conv6(x).sigmoid().squeeze(-1)
        x7 = self.conv7(x).sigmoid().squeeze(-1)

        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x4 = x4.view(x4.shape[0], -1)
        x5 = x5.view(x5.shape[0], -1)
        x6 = x6.view(x6.shape[0], -1)
        x7 = x7.view(x7.shape[0], -1)

        concat = torch.cat([x2,x3,x4,x5,x6,x7], dim=1)

        feature_loc = self.featloc(concat)

        out_cnn = self.mlp1(feature_loc)

        out = self.t(out_cnn) * self.mlp2(out_cnn) + (1 - self.t(out_cnn)) * out_cnn

        return out

class mimick_cnn4(nn.Module):
    def __init__(self, embedding, char_max_len=15, char_emb_dim=300, emb_dim=300, num_feature=100, classif=200, random=False, asc=False):
        super(mimick_cnn4, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.conv2 = nn.Conv2d(1, num_feature, (2, char_emb_dim))
        self.conv3 = nn.Conv2d(1, num_feature, (3, char_emb_dim))
        self.conv4 = nn.Conv2d(1, num_feature, (4, char_emb_dim))
        self.conv5 = nn.Conv2d(1, num_feature, (5, char_emb_dim))
        self.conv6 = nn.Conv2d(1, num_feature, (6, char_emb_dim))
        self.conv7 = nn.Conv2d(1, num_feature, (7, char_emb_dim))

        self.classif = nn.Sequential(
            nn.Linear(num_feature*48, classif),
            nn.LogSoftmax()
        )

        self.regres = nn.Sequential(
            nn.Linear(classif, emb_dim),
            nn.Hardtanh(min_val=-3, max_val=3)
        )

    def forward(self, inputs):
        x2 = self.conv2(inputs).relu().squeeze(-1)
        x3 = self.conv3(inputs).relu().squeeze(-1)
        x4 = self.conv4(inputs).relu().squeeze(-1)
        x5 = self.conv5(inputs).relu().squeeze(-1)
        x6 = self.conv6(inputs).relu().squeeze(-1)
        x7 = self.conv7(inputs).relu().squeeze(-1)


        x2_max = F.max_pool1d(x2, 2).squeeze(-1)
        x3_max = F.max_pool1d(x3, 2).squeeze(-1)
        x4_max = F.max_pool1d(x4, 2).squeeze(-1)
        x5_max = F.max_pool1d(x5, 2).squeeze(-1)
        x6_max = F.max_pool1d(x6, 2).squeeze(-1)
        x7_max = F.max_pool1d(x7, 2).squeeze(-1)


        maxpoolcat = torch.cat([x2_max, x3_max, x4_max, x5_max, x6_max, x7_max], dim=2).view(inputs.size(0), -1)

        c = self.classif(maxpoolcat)

        out = self.regres(c)

        return out

# %% [markdown]
# # Train

# %%
for args_embedding in ['word2vec']:
    for args_model in ['cnn']:
        for pca in [None] + np.arange(0.1,1,0.2).tolist(): # 
            embedding_size = {
                'polyglot' : 64,
                'word2vec' : 300,
                'dict2vec' :100
            }
            pca_components = int(np.floor(pca*embedding_size[args_embedding])) if pca else None
            args_max_epoch = 40
            args_run = 1
            args_save = True
            args_load = False
            args_lang = "en"
            args_lr = 0.1
            args_charlen = 20
            args_charembdim = 300
            args_local = True
            args_loss_fn = "mse"
            args_dropout = 0
            args_bsize = 30
            args_epoch = 0
            args_asc = False
            args_quiet = True
            args_init_weight = False
            args_shuffle = False
            args_nesterov = False
            args_loss_reduction = False
            args_num_feature = 50
            args_weight_decay = 0
            args_momentum = 0
            args_multiplier = 1
            args_classif = 200
            args_neighbor = 5
            args_seed = 64

            # %%
            #@title Training
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torch.optim as optim
            from torch.autograd import Variable, gradcheck
            from torch.utils.data import SubsetRandomSampler, DataLoader
            # from torch.utils.tensorboard import SummaryWriter
            from datetime import datetime, timezone, timedelta
            timezone_offset = 7.0  # Pacific Standard Time (UTC−08:00)
            tzinfo = timezone(timedelta(hours=timezone_offset))
            curr_dt = datetime.now(tzinfo)
            # writer = SummaryWriter()
            if pca:
                wab_name = f"{args_model}{args_embedding}{args_loss_fn}{pca}_{curr_dt.year-2000}{curr_dt.month:02}{curr_dt.day:02}{curr_dt.hour:02}{curr_dt.minute:02}"
            else:
                wab_name = f"{args_model}{args_embedding}{args_loss_fn}_{curr_dt.year-2000}{curr_dt.month:02}{curr_dt.day:02}{curr_dt.hour:02}{curr_dt.minute:02}"
            cfg_mimick = {
                    "run":int(args_run),
                    "char_emb_dim":int(args_charembdim),
                    "char_max_len":int(args_charlen),
                    "char_emb_ascii":args_asc,
                    "random_seed":int(args_seed),
                    "shuffle_dataset":args_shuffle,
                    "neighbor":int(args_neighbor),
                    "validation_split":.8,
                    "batch_size":int(args_bsize),
                    "max_epoch":int(args_max_epoch),
                    "learning_rate":float(args_lr),
                    "weight_decay":float(args_weight_decay),
                    "momentum":float(args_momentum),
                    "multiplier":float(args_multiplier),
                    "classif":int(args_classif),
                    "model":args_model,
                    "loss_fn":args_loss_fn,
                    "loss_reduction":args_loss_reduction,
                    "pca_components":pca_components
                }
            pprint.pp(cfg_mimick)
            wandb.init(
                # Set the project where this run will be logged
                project="Mimick",
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=wab_name,
                config=cfg_mimick
            )

            import numpy as np
            import math

            # from model import *
            # from charembedding import Char_embedding
            # from wordembedding import Word_embedding

            # import argparse

            from tqdm import trange, tqdm
            import os

            from distutils.dir_util import copy_tree

            def cosine_similarity(tensor1, tensor2, neighbor=5):
                '''
                Calculating cosine similarity for each vector elements of
                tensor 1 with each vector elements of tensor 2

                Input:

                tensor1 = (torch.FloatTensor) with size N x D
                tensor2 = (torch.FloatTensor) with size M x D
                neighbor = (int) number of closest vector to be returned

                Output:

                (distance, neighbor)
                '''
                tensor1_norm = torch.norm(tensor1, 2, 1)
                tensor2_norm = torch.norm(tensor2, 2, 1)
                tensor1_dot_tensor2 = torch.mm(tensor2, torch.t(tensor1)).t()

                divisor = [t * tensor2_norm for t in tensor1_norm]

                divisor = torch.stack(divisor)

                # result = (tensor1_dot_tensor2/divisor).data.cpu().numpy()
                result = (tensor1_dot_tensor2/divisor.clamp(min=1.e-09)).data.cpu()
                d, n = torch.sort(result, descending=True)
                n = n[:, :neighbor]
                d = d[:, :neighbor]
                return d, n

            def init_weights(m):
                if type(m) == nn.Linear or type(m) == nn.Conv2d:
                    m.weight.data.fill_(0.01)
                    m.bias.data.fill_(0.01)

            def pairwise_distances(x, y=None, multiplier=1., loss=False, neighbor=5):
                '''
                Input:

                x is a Nxd matrix
                y is an optional Mxd matirx

                Output:

                dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.

                i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
                '''
                x_norm = (x**2).sum(1).view(-1, 1)
                if y is not None:
                    y *= multiplier
                    y_norm = (y**2).sum(1).view(1, -1)
                else:
                    y = x
                    y_norm = x_norm.view(1, -1)
                if loss:
                    result = F.pairwise_distance(x, y)
                    return result
                else:
                    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
                    d, n = torch.sort(dist, descending=False)
                    n = n[:, :neighbor]
                    d = d[:, :neighbor]
                    return d, n

            def decaying_alpha_beta(epoch=0, loss_fn='cosine'):
                # decay = math.exp(-float(epoch)/200)
                if loss_fn == 'cosine':
                    alpha = 1
                    beta = 0.5
                else:
                    alpha = 0.5
                    beta = 1
                return alpha, beta

            cloud_dir = '/content/gdrive/My Drive/train_dropout/'
            if pca:
                saved_model_path = 'trained_model_%s_%s_%s_%s_%s' % (args_lang, args_model, args_embedding, args_loss_fn, pca_components)
            else:
                saved_model_path = 'trained_model_%s_%s_%s_%s' % (args_lang, args_model, args_embedding, args_loss_fn)
            logger_dir = '%s/logs/run%s/' % (saved_model_path, args_run)
            logger_val_dir = '%s/logs/val-run%s/' % (saved_model_path, args_run)

            if not args_local:
                # logger_dir = cloud_dir + logger_dir
                saved_model_path = cloud_dir + saved_model_path

            # *Device configuration
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # *Parameters
            run = int(args_run)
            char_emb_dim = int(args_charembdim)
            char_max_len = int(args_charlen)
            random_seed = int(args_seed)
            shuffle_dataset = args_shuffle
            neighbor = int(args_neighbor)
            validation_split = .8
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

            # *Hyperparameter
            batch_size = int(args_bsize)
            max_epoch = int(args_max_epoch)
            learning_rate = float(args_lr)
            weight_decay = float(args_weight_decay)
            momentum = float(args_momentum)
            multiplier = float(args_multiplier)
            classif = int(args_classif)

            val_batch_size = 30

            char_embed = Char_embedding(char_emb_dim, char_max_len, asc=args_asc, random=True, device=device)

            dataset_we = Word_embedding(lang=args_lang, embedding=args_embedding, pca=pca_components)
            print(args_embedding)
            emb_dim = dataset_we.emb_dim

            dataset_size = len(dataset_we)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))


            # if shuffle_dataset:
            np.random.shuffle(indices)

            #* Creating PT data samplers and loaders:
            train_indices, val_indices = indices[:split], indices[split:]

            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(dataset_we, batch_size=batch_size,
                                            sampler=train_sampler)
            validation_loader = DataLoader(dataset_we, batch_size=val_batch_size,
                                            sampler=valid_sampler)

            #* Initializing model
            if args_model == 'lstm':
                model = mimick(char_embed.embed, char_embed.char_emb_dim, dataset_we.emb_dim, int(args_num_feature))
            elif args_model == 'cnn2':
                model = mimick_cnn2(
                    embedding=char_embed.embed,
                    char_max_len=char_embed.char_max_len,
                    char_emb_dim=char_embed.char_emb_dim,
                    emb_dim=emb_dim,
                    num_feature=int(args_num_feature),
                    random=False, asc=args_asc)
            elif args_model == 'cnn':
                model = mimick_cnn(
                    embedding=char_embed.embed,
                    char_max_len=char_embed.char_max_len,
                    char_emb_dim=char_embed.char_emb_dim,
                    emb_dim=emb_dim,
                    num_feature=int(args_num_feature),
                    random=False, asc=args_asc)
            elif args_model == 'cnn3':
                model = mimick_cnn3(
                    embedding=char_embed.embed,
                    char_max_len=char_embed.char_max_len,
                    char_emb_dim=char_embed.char_emb_dim,
                    emb_dim=emb_dim,
                    num_feature=int(args_num_feature),
                    mtp=multiplier,
                    random=False, asc=args_asc)
            elif args_model == 'cnn4':
                model = mimick_cnn4(
                    embedding=char_embed.embed,
                    char_max_len=char_embed.char_max_len,
                    char_emb_dim=char_embed.char_emb_dim,
                    emb_dim=emb_dim,
                    num_feature=int(args_num_feature),
                    classif=classif,
                    random=False, asc=args_asc)
            else:
                model = None

            model.to(device)

            if args_loss_fn == 'mse':
                if args_loss_reduction:
                    criterion = nn.MSELoss(reduction='none')
                else:
                    criterion = nn.MSELoss()
            else:
                criterion = nn.CosineSimilarity()

            if run < 1:
                args_run = '1'

            if run != 1:
                args_load = True

            if args_load:
                model.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, args_model)))
            elif not os.path.exists(saved_model_path):
                os.makedirs(saved_model_path)

            word_embedding = dataset_we.embedding_vectors.to(device)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=args_nesterov)

            if args_init_weight: model.apply(init_weights)

            step = 0
            # *Training
            print(pca)
            for epoch in trange(int(args_epoch), max_epoch, total=max_epoch, initial=int(args_epoch)):
                # conv2weight = model.conv2.weight.data.clone()
                # mlpweight = model.mlp[2].weight.data.clone()
                sum_loss = 0.
                for it, (X, y) in enumerate(train_loader):
                    model.train()
                    model.zero_grad()
                    words = dataset_we.idxs2words(X)
                    idxs = char_embed.char_split(words).to(device)
                    if args_model != 'lstm': idxs = idxs.unsqueeze(1)
                    inputs = Variable(idxs).to(device) # (length x batch x char_emb_dim)
                    target = Variable(y*multiplier).squeeze().to(device) # (batch x word_emb_dim)

                    output = model.forward(inputs) # (batch x word_emb_dim)
                    # print(output.shape)
                    # print(target.shape)

                    loss = criterion(output, target) if args_loss_fn == 'mse' else (1-criterion(output, target)).mean()

                    # ##################
                    # Tensorboard
                    # ##################
                    sum_loss += loss.item() if not args_loss_reduction else loss.mean().item()

                    if not args_loss_reduction:
                        loss.backward()
                    else:
                        loss = loss.mean(0)
                        for i in range(len(loss)-1):
                            loss[i].backward(retain_graph=True)

                        loss[len(loss)-1].backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    if not args_quiet:
                        if it % int(dataset_size/(batch_size*5)) == 0:
                            tqdm.write('loss = %.4f' % loss.mean())
                            model.eval()
                            random_input = np.random.randint(len(X))

                            words = dataset_we.idx2word(X[random_input]) # list of words

                            distance, nearest_neighbor = cosine_similarity(output[random_input].detach().unsqueeze(0), word_embedding, neighbor=neighbor)

                            loss_dist = torch.dist(output[random_input], target[random_input]*multiplier)
                            tqdm.write('%d %.4f | ' % (step, loss_dist.item()) + words + '\t=> ' + dataset_we.idxs2sentence(nearest_neighbor[0]))
                            model.train()
                            tqdm.write('')
                # writer.add_scalar('Loss/train', sum_loss, epoch)
                wandb.log({"train_loss": sum_loss / len(train_indices)})
                model.eval()

                ############################
                # SAVING TRAINED MODEL
                ############################

                if not args_local:
                    copy_tree(logger_dir, cloud_dir+logger_dir)

                torch.save(model.state_dict(), f'{saved_model_path}/{args_model}.pth')

                val_loss = 0.
                for it, (X, target) in enumerate(validation_loader):
                    words = dataset_we.idxs2words(X)
                    idxs = char_embed.char_split(words).to(device)
                    if args_model != 'lstm': idxs = idxs.unsqueeze(1)
                    inputs = Variable(idxs) # (length x batch x char_emb_dim)
                    if pca:
                        target = dataset_we.pca.inverse_transform(target).to(device) # (batch x word_emb_dim)
                    else:
                        target = target.to(device) # (batch x word_emb_dim)

                    model.zero_grad()

                    output = model.forward(inputs) # (batch x word_emb_dim)
                    if pca:
                        output = dataset_we.pca.inverse_transform(output).to(device) # (batch x word_emb_dim)

                    loss = criterion(output, target) if args_loss_fn == 'mse' else (1-criterion(output, target)).mean()
                    val_loss += loss.item() if not args_loss_reduction else loss.mean().item()

                    if not args_quiet:
                        if it < 1:
                            distance, nearest_neighbor = cosine_similarity(output, word_embedding, neighbor=neighbor)

                            for i, word in enumerate(X):
                                if i >= 1: break
                                loss_dist = torch.dist(output[i], target[i])

                                tqdm.write('%.4f | %s \t=> %s' % (loss_dist.item(), dataset_we.idx2word(word), dataset_we.idxs2sentence(nearest_neighbor[i])))
                # writer.add_scalar('Loss/Val', val_loss, epoch)
                wandb.log({"val_loss": val_loss / len(val_indices)})

                if not args_quiet: print('total loss = %.8f' % val_loss)

            #@title Testing Similarities
            model.eval()

            # *Evaluating
            words = 'MCT McNeally Vercellotti Secretive corssing flatfish compartmentalize pesky lawnmower developiong hurtling expectedly'.split()
            # words += args_words

            inputs = char_embed.char_split(words)

            embedding = dataset_we.embedding_vectors.to(device)
            inputs = inputs.to(device) # (length x batch x char_emb_dim)
            if args_model != 'lstm': inputs = inputs.unsqueeze(1)
            if pca:
                output = dataset_we.pca.inverse_transform(model.forward(inputs)) # (batch x word_emb_dim)
            else:
                output = model.forward(inputs) # (batch x word_emb_dim)

            cos_dist, nearest_neighbor = cosine_similarity(output, embedding, neighbor)

            sim_table = []

            for i, word in enumerate(words):
                sim_table += [[torch.mean(cos_dist[i]), word, dataset_we.idxs2sentence(nearest_neighbor[i])]]
                # print('%.4f | ' % torch.mean(cos_dist[i]) + word + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[i]))
            my_table = wandb.Table(columns=["Cosine Similarity", "Word", "Similar Words"], data=sim_table)
            wandb.log({"Similarity Test": my_table})
            wandb.finish()

            # %% [markdown]
            # # Downstream Task

            # %%
            #@title POStag Model
            import nltk
            import re

            if not os.path.exists('brown.txt'):
                response = requests.get('https://raw.githubusercontent.com/yonathansantosa/Mimick/master/tagset/brown.txt')
                with open("brown.txt", mode="wb") as file:
                    file.write(response.content)
            if not os.path.exists('upenn.txt'):
                response = requests.get('https://raw.githubusercontent.com/yonathansantosa/Mimick/master/tagset/upenn.txt')
                with open("upenn.txt", mode="wb") as file:
                    file.write(response.content)
            np.random.seed(0)

            class Tagset:
                def __init__(self, tagset='brown'):
                    self.itot = {}
                    self.toti = {}
                    with open ('%s.txt' % tagset, "r") as myfile:
                        data=myfile.readlines()
                        sent = "".join([d for d in data])
                        processed = re.findall(r"(.*):", sent)
                        for i, tag in enumerate(processed):
                            self.toti[tag] = i

                        for i, tag in enumerate(processed):
                            self.itot[i] = tag

                def __len__(self):
                    return len(self.itot)

                def idx2tag(self, idx):
                    return self.itot[idx]

                def tag2idx(self, tag):
                    return self.toti[tag]

            class Postag:
                def __init__(self, word_embed, corpus='brown', tagset='brown', device='cuda'):
                    if corpus == 'brown':
                        nltk.download('brown')
                        from nltk.corpus import brown as corpus
                    self.word_embed = word_embed
                    self.tagged_words = corpus.tagged_words(tagset=tagset)
                    self.tagged_sents = corpus.tagged_sents(tagset=tagset)
                    self.tagset = Tagset(tagset=tagset)
                    new_itot = {}
                    new_toti = {}
                    self.count_bin = torch.zeros(len(self.tagset))
                    self.idxs = torch.zeros(1)
                    self.device = device

                    for word, tag in self.tagged_words:
                        if tag in self.tagset.toti:
                            self.count_bin[self.tagset.tag2idx(tag)] += 1
                        else:
                            self.count_bin[self.tagset.tag2idx('UNK')] += 1

                    _, self.idxs = torch.sort(self.count_bin, descending=True)

                    for it, i in enumerate(self.idxs):
                        new_itot[it] = self.tagset.itot[int(i)]
                        new_toti[new_itot[it]] = it

                    self.tagset.toti = new_toti
                    self.tagset.itot = new_itot


                def __len__(self):
                    return len(self.tagged_sents)


                def __getitem__(self, index):
                    length = len(self.tagged_sents[index])
                    word = []
                    tag = []

                    if length-5 <= 0:
                        for i in range(length):
                            w, t = self.tagged_sents[index][i]
                            word += [self.word_embed.word2idxs(w)]

                            if t in self.tagset.toti:
                                tag_id = self.tagset.tag2idx(t)
                            else:
                                tag_id = self.tagset.tag2idx('UNK')

                            tag += [tag_id]

                        for i in range(length, 5):
                            word += [self.word_embed.word2idxs('<pad>')]
                            tag_id = self.tagset.tag2idx('UNK')

                            tag += [tag_id]

                    else:
                        start_index = np.random.randint(0, length-5)
                        for i in range(start_index, start_index+5):
                            w, t = self.tagged_sents[index][i]
                            if t in self.tagset.toti:
                                tag_id = self.tagset.tag2idx(t)
                            else:
                                tag_id = self.tagset.tag2idx('UNK')
                            word += [self.word_embed.word2idxs(w)]
                            tag += [tag_id]

                    # word_emb = self.word_embed.word_embedding(torch.tensor(word).to(self.device))
                    # return (word_emb, torch.LongTensor(tag).view(len(tag), 1), torch.LongTensor(word).view(len(word), 1))
                    return (torch.vstack(word), torch.LongTensor(tag))

            class Postag_word:
                def __init__(self, word_embed, char_embed, corpus='brown', tagset='brown'):
                    if corpus == 'brown':
                        nltk.download('brown')
                        from nltk.corpus import brown as corpus
                    self.char_embed = char_embed
                    self.tagged_words = corpus.tagged_words(tagset=tagset)
                    self.tagged_sents = corpus.tagged_sents(tagset=tagset)
                    self.tagset = Tagset(tagset=tagset)
                    new_itot = {}
                    new_toti = {}
                    self.word_embed = word_embed
                    self.count_bin = torch.zeros(len(self.tagset))
                    self.idxs = torch.zeros(1)

                    for _, tag in self.tagged_words:
                        if tag in self.tagset.toti:
                            self.count_bin[self.tagset.tag2idx(tag)] += 1
                        else:
                            self.count_bin[self.tagset.tag2idx('UNK')] += 1

                    _, self.idxs = torch.sort(self.count_bin, descending=True)

                    for it, i in enumerate(self.idxs):
                        new_itot[it] = self.tagset.itot[int(i)]
                        new_toti[new_itot[it]] = it

                    self.tagset.toti = new_toti
                    self.tagset.itot = new_itot


                def __len__(self):
                    return len(self.tagged_sents)


                def __getitem__(self, index):
                    word, tag = self.tagged_words[index]

                    # if word in self.word_embed.stoi:
                    #     w_idx = self.word_embed.stoi[word]
                    # else:
                    #     w_idx = self.word_embed.stoi['</s>']

                    w_c_idx = self.char_embed.word2idxs(word)
                    if tag in self.tagset.toti:
                        tag_id = self.tagset.tag2idx(tag)
                    else:
                        tag_id = self.tagset.tag2idx('UNK')

                    try:
                        w_idx = self.word_embed.stoi[word]
                    except:
                        pass

                    return (torch.LongTensor(w_idx), torch.LongTensor(w_c_idx), torch.LongTensor(tag_id))

            class Postagger(nn.Module):
                def __init__(self, seq_length, emb_dim, hidden_size, output_size):
                    super(Postagger, self).__init__()
                    self.hidden_size = hidden_size
                    self.seq_length = seq_length
                    self.lstm = nn.LSTM(emb_dim, self.hidden_size, 1, bidirectional=True, batch_first=True)
                    self.lstm.flatten_parameters()
                    self.mlp = nn.Sequential(
                        nn.Linear(self.hidden_size, output_size),
                        nn.LogSoftmax(dim=2),
                    )


                def forward(self, inputs):
                    self.lstm.flatten_parameters()
                    out, _ = self.lstm(inputs)

                    output = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]

                    out = self.mlp(output)

                    return out

            class Postagger_adaptive(nn.Module):
                def __init__(self, seq_length, emb_dim, hidden_size, output_size):
                    super(Postagger_adaptive, self).__init__()
                    self.hidden_size = hidden_size
                    self.seq_length = seq_length
                    self.lstm = nn.LSTM(emb_dim, self.hidden_size, 1, bidirectional=True, batch_first=True)
                    self.lstm.flatten_parameters()
                    self.out = nn.AdaptiveLogSoftmaxWithLoss(hidden_size, output_size, cutoffs=[round(output_size/5),2*round(output_size/5)], div_value=4)

                def forward(self, inputs, targets):
                    self.lstm.flatten_parameters()
                    out, _ = self.lstm(inputs)

                    output = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
                    output = output.contiguous().view(output.shape[0]*output.shape[1], output.shape[2])
                    targets = targets.view(targets.shape[0]*targets.shape[1])

                    return self.out(output, targets)

                def validation(self, inputs, targets):
                    self.lstm.flatten_parameters()
                    out, _ = self.lstm(inputs)

                    output = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
                    output = output.contiguous().view(output.shape[0] * output.shape[1], -1)
                    targets = targets.view(targets.shape[0]*targets.shape[1])

                    prediction = self.out.predict(output)
                    _, loss = self.out(output, targets)

                    return prediction, float(loss.cpu())

            # %%
            # import argparse

            args_max_epoch = 50
            args_run = 1
            args_save = True
            args_load = False
            args_lang = "en"
            # args_model = "lstm"
            args_lr = 0.1
            args_charlen = 20
            args_charembdim = 300
            # args_embedding = "word2vec"
            args_local = True
            args_loss_fn = "mse"
            args_dropout = 0
            args_bsize = 64
            args_epoch = 0
            args_asc = False
            args_quiet = True
            args_init_weight = False
            args_shuffle = False
            args_nesterov = False
            args_loss_reduction = False
            args_num_feature = 50
            args_weight_decay = 0
            args_momentum = 0
            args_multiplier = 1
            args_classif = 200
            args_neighbor = 5
            args_seq_len = 5
            # %%
            #@title POStag Train
            import shutil
            from distutils.dir_util import copy_tree

            def init_weights(m):
                if type(m) == nn.Linear or type(m) == nn.Conv2d:
                    m.weight.data.fill_(0.01)
                    m.bias.data.fill_(0.01)

            # *Device configuration
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            cloud_dir = '/content/gdrive/My Drive/train_dropout/'
            if pca:
                saved_model_path = 'trained_model_%s_%s_%s_%s_%s' % (args_lang, args_model, args_embedding, args_loss_fn, pca_components)
                saved_postag_path = 'trained_model_%s_%s_%s_%s_%s_postag' % (args_lang, args_model, args_embedding, args_loss_fn, pca_components)
            else:
                saved_postag_path = 'trained_model_%s_%s_%s_%s_postag' % (args_lang, args_model, args_embedding, args_loss_fn)
                saved_model_path = 'trained_model_%s_%s_%s_%s' % (args_lang, args_model, args_embedding, args_loss_fn)
            logger_dir = '%s/logs/run%s/' % (saved_postag_path, args_run)
            logger_val_dir = '%s/logs/val-run%s/' % (saved_postag_path, args_run)
            logger_val_cosine_dir = '%s/logs/val-cosine-run%s/' % (saved_postag_path, args_run)

            if not args_local:
                # logger_dir = cloud_dir + logger_dir
                saved_model_path = cloud_dir + saved_model_path
                saved_postag_path = cloud_dir + saved_postag_path

            if pca:
                wab_name = f"{args_model}{args_embedding}{pca}_{curr_dt.year-2000}{curr_dt.month:02}{curr_dt.day:02}{curr_dt.hour:02}{curr_dt.minute:02}_POStag"
            else:
                wab_name = f"{args_model}{args_embedding}_{curr_dt.year-2000}{curr_dt.month:02}{curr_dt.day:02}{curr_dt.hour:02}{curr_dt.minute:02}_POStag"
            cfg_postag = {
                    "run":int(args_run),
                    "char_emb_dim":int(args_charembdim),
                    "char_max_len":int(args_charlen),
                    "char_emb_ascii":args_asc,
                    "shuffle_dataset":args_shuffle,
                    "neighbor":int(args_neighbor),
                    "validation_split":.8,
                    "batch_size":int(args_bsize),
                    "max_epoch":int(args_max_epoch),
                    "learning_rate":float(args_lr),
                    "weight_decay":float(args_weight_decay),
                    "momentum":float(args_momentum),
                    "multiplier":float(args_multiplier),
                    "classif":int(args_classif),
                    "model":args_model,
                    "loss_fn":args_loss_fn,
                    "loss_reduction":args_loss_reduction,
                    "seq_len":int(args_seq_len)
                }
            wandb.init(
                # Set the project where this run will be logged
                project="Mimick",
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=wab_name,
                config=cfg_postag
            )
            pprint.pp(cfg_postag)


            # *Parameters
            char_emb_dim = int(args_charembdim)
            char_max_len = int(args_charlen)
            random_seed = 64
            shuffle_dataset = args_shuffle
            validation_split = .8
            neighbor = int(args_neighbor)
            seq_len = int(args_seq_len)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

            # *Hyperparameter
            batch_size = int(args_bsize)
            val_batch_size = 30
            max_epoch = int(args_max_epoch)
            learning_rate = float(args_lr)
            weight_decay = float(args_weight_decay)
            momentum = float(args_momentum)
            multiplier = float(args_multiplier)
            classif = int(args_classif)

            char_embed = Char_embedding(char_emb_dim, char_max_len, asc=args_asc, random=True, device=device)
            # char_embed.embed.load_state_dict(torch.load('%s/charembed.pth' % saved_model_path))
            # char_embed.embed.eval()
            dataset = Postag(char_embed)

            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))

            if shuffle_dataset:
                np.random.seed(random_seed)
                np.random.shuffle(indices)

            #* Creating PT data samplers and loaders:
            train_indices, val_indices = indices[:split], indices[split:]

            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(dataset, batch_size=batch_size,
                                            sampler=train_sampler)
            validation_loader = DataLoader(dataset, batch_size=val_batch_size,
                                            sampler=valid_sampler)

            #* Initializing model
            # print(args_embedding)
            # print(args_embedding)
            # word_emb = Word_embedding(lang=args_lang, embedding=args_embedding, pca=pca)
            # emb_dim = word_emb.emb_dim

            if args_model == 'lstm':
                model = mimick(char_embed.embed, char_embed.char_emb_dim, emb_dim, int(args_num_feature))
            elif args_model == 'cnn2':
                model = mimick_cnn2(
                    embedding=char_embed.embed,
                    char_max_len=char_embed.char_max_len,
                    char_emb_dim=char_embed.char_emb_dim,
                    emb_dim=emb_dim,
                    num_feature=int(args_num_feature),
                    random=False, asc=args_asc)
            elif args_model == 'cnn':
                model = mimick_cnn(
                    embedding=char_embed.embed,
                    char_max_len=char_embed.char_max_len,
                    char_emb_dim=char_embed.char_emb_dim,
                    emb_dim=emb_dim,
                    num_feature=int(args_num_feature),
                    random=False, asc=args_asc)
            elif args_model == 'cnn3':
                model = mimick_cnn3(
                    embedding=char_embed.embed,
                    char_max_len=char_embed.char_max_len,
                    char_emb_dim=char_embed.char_emb_dim,
                    emb_dim=emb_dim,
                    num_feature=int(args_num_feature),
                    mtp=multiplier,
                    random=False, asc=args_asc)
            elif args_model == 'cnn4':
                model = mimick_cnn4(
                    embedding=char_embed.embed,
                    char_max_len=char_embed.char_max_len,
                    char_emb_dim=char_embed.char_emb_dim,
                    emb_dim=emb_dim,
                    num_feature=int(args_num_feature),
                    classif=classif,
                    random=False, asc=args_asc)
            else:
                model = None

            model.to(device)
            model.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, args_model)))
            model.eval()

            postagger = Postagger_adaptive(seq_len, embedding_size[args_embedding], 20, len(dataset.tagset)).to(device)
            # postagger = Postagger(seq_len, emb_dim, 20, len(dataset.tagset)).to(device)

            if args_load:
                postagger.load_state_dict(torch.load('%s/postag.pth' % (saved_postag_path)))

            optimizer = optim.SGD(postagger.parameters(), lr=learning_rate, momentum=momentum, nesterov=args_nesterov)
            criterion = nn.NLLLoss()

            if args_init_weight: postagger.apply(init_weights)
            step = 0

            #* Training
            for epoch in trange(int(args_epoch), max_epoch, total=max_epoch, initial=int(args_epoch)):
                loss_item = 0.
                for it, (X, y) in enumerate(train_loader):
                    postagger.zero_grad()
                    if args_model == 'lstm':
                        inputs = X.view(X.shape[0]*seq_len, X.shape[2]).to(device)
                    else:
                        inputs = X.view(X.shape[0]*seq_len, 1, -1).to(device)
                    emb_pred = model.forward(inputs)
                    if pca:
                        emb_pred = dataset_we.pca.inverse_transform(emb_pred)
                    w_embedding = Variable(emb_pred.view(X.shape[0], seq_len, -1), requires_grad=True).to(device) # (batch x sent_length x word_emb_dim)
                    target = Variable(y).to(device)
                    # output = postagger.forward(w_embedding, target).permute(0, 2, 1)
                    output, loss = postagger.forward(w_embedding, target)

                    # loss = criterion(output, target)

                    # ##################
                    # Tensorboard
                    # ##################
                    # loss_item = loss.item() if not args_loss_reduction else loss.mean().item()
                    # info = {
                    #     'loss-Train-%s-postag-run%s' % (args_model, args_run) : loss_item,
                    # }

                    # step += 1
                    # if args_run != 0:
                    #     for tag, value in info.items():
                    #         logger.scalar_summary(tag, value, step)


                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    loss_item += loss.detach().cpu().item() if not args_loss_reduction else loss.detach().cpu().mean().item()

                wandb.log({"train_loss": loss_item})

                # if not args_local:
                #     copy_tree(logger_dir, cloud_dir+logger_dir)

                if not os.path.exists(saved_postag_path):
                    os.makedirs(saved_postag_path)

                torch.save(postagger.state_dict(), '%s/postag.pth' % (saved_postag_path))
                if not args_quiet: tqdm.write('%d | %.4f ' % (epoch, loss_item))

                #* Validation
                postagger.eval()
                validation_loss = 0.
                accuracy = 0.
                for it, (X, y) in enumerate(validation_loader):
                    if args_model == 'lstm':
                        inputs = X.view(X.shape[0]*seq_len, X.shape[2]).to(device)
                    else:
                        inputs = X.view(X.shape[0]*seq_len, 1, -1).to(device)
                    
                    emb_pred = model.forward(inputs)
                    if pca:
                        emb_pred = dataset_we.pca.inverse_transform(emb_pred)
                    w_embedding = Variable(emb_pred.view(X.shape[0], seq_len, -1), requires_grad=False).to(device) # (batch x sent_length x word_emb_dim)
                    target = Variable(y).to(device)
                    # output = postagger.forward(w_embedding).permute(0, 2, 1)
                    output, validation_loss = postagger.validation(w_embedding, target)
                    # output_tag = postagger.predict(output.view(X.shape[0], seq_len))
                    output_tag = output.view(X.shape[0], seq_len)
                    correct = (output_tag == target).sum()/(len(val_indices)*seq_len)
                    accuracy += correct
                    # validation_loss += criterion(output, target)*X.shape[0]/(len(val_indices))
                    if not args_quiet:
                        if it == 0:
                            tag = output_tag[0]
                            # output_tag = postagger.predict(output.view(X.shape[0], seq_len)[0])
                            for i in range(len(X[0])):
                                word_idx = X[0][i].numpy()
                                word = char_embed.clean_idxs2word(word_idx)
                                tg = dataset.tagset.idx2tag(int(tag[i].cpu()))
                                tgt = dataset.tagset.idx2tag(int(y[0][i]))
                                tqdm.write('(%s, %s) => %s' % (word, tgt, tg))
                if not args_quiet: tqdm.write('accuracy = %.4f' % accuracy)
                wandb.log({"validation_accuracy": accuracy})
                # info_val = {
                #     'loss-Train-%s-postag-run%s' % (args_model, args_run) : validation_loss
                # }

                # if args_run != 0:
                #     for tag, value in info_val.items():
                #         logger_val.scalar_summary(tag, validation_loss, step)

                if not args_quiet: tqdm.write('val_loss %.4f ' % validation_loss)

                postagger.train()

            postagger.eval()

            accuracy = 0.
            for it, (X, y) in enumerate(validation_loader):
                if args_model == 'lstm':
                    inputs = X.view(X.shape[0]*seq_len, X.shape[2]).to(device)
                else:
                    inputs = X.view(X.shape[0]*seq_len, 1, -1).to(device)
                
                emb_pred = model.forward(inputs)
                if pca:
                    emb_pred = dataset_we.pca.inverse_transform(emb_pred)
                w_embedding = Variable(emb_pred.view(X.shape[0], 5, -1), requires_grad=False).to(device) # (batch x sent_length x word_emb_dim)
                target = Variable(y).to(device)
                # output = postagger.forward(w_embedding).permute(0, 2, 1)
                output, _ = postagger.validation(w_embedding, target)
                # output_tag = postagger.predict(output.view(X.shape[0], seq_len))
                output_tag = output.view(X.shape[0], seq_len)
                correct = float((output_tag == target).sum())/(len(val_indices)*seq_len)
                accuracy += correct
                if it <= 3:
                    tag = output_tag[0]
                    for i in range(len(X[0])):
                        word_idx = X[0][i].numpy()
                        word = char_embed.clean_idxs2word(word_idx)
                        tg = dataset.tagset.idx2tag(int(tag[i].cpu()))
                        tgt = dataset.tagset.idx2tag(int(y[0][i]))
                        tqdm.write('(%s, %s) => %s' % (word, tgt, tg))
                    tqdm.write('\n')
            print('accuracy = %.4f' % accuracy)
            wandb.summary["postag_validation_accuracy"] = accuracy
            wandb.finish()