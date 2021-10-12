import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Arch1(nn.Module):
    def __init__(self, emb_num, emb_dim):
        super().__init__()

        self.embedding = nn.Embedding(emb_num, emb_dim)
        self.linear = nn.Linear(2 * emb_dim, 1)
    
    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).

        :param fine_tune: Fine-tune?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
    
    def forward(self, text, tag, text_length):
        emb_text = self.embedding(text)  # (batch_size, seq_len, emb_dim)
        emb_tag = self.embedding(tag)  # (batch_size, topic_len, emb_dim)

        mean_text = emb_text.mean(dim=1)  # (batch_size, emb_dim)
        mean_tag = emb_tag.mean(dim=1)

        concatenates = torch.cat([mean_text, mean_tag], dim=1)  # (batch_size, 2*emb_dim)
        out = self.linear(concatenates)  # (batch_size, 1)

        return torch.sigmoid(out)


class Arch2(nn.Module):
    def __init__(self, emb_num, emb_dim):
        super(). __init__()
        dropout = 0.5
        self.embedding = nn.Embedding(emb_num, emb_dim)
        self.lstm = nn.LSTM(emb_dim, emb_dim, 
                            dropout = dropout, batch_first = True)
        self.linear = nn.Linear(2 * emb_dim, 1)
    
    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).

        :param fine_tune: Fine-tune?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune    
    
    def forward(self, text, tag, text_length):

        text_length, word_sort_ind = text_length.sort(dim=0, descending=True)
        text = text[word_sort_ind]
        tag = tag[word_sort_ind]
        batch_size = text.size(0)

        emb_text = self.embedding(text)  # (batch_size, seq_len, emb_dim)
        emb_tag = self.embedding(tag)  # (batch_size, topic_len, emb_dim)

        #Pack padded sequence(这里需要验证一下pack后输入lstm得到的hidden state是什么)
        w = pack_padded_sequence(emb_text, list(text_length),
                                 batch_first=True)
        w, (final_hidden_state, _) = self.lstm(w)  # (1, batch_size, emb_dim)
        #last_hidden = last_hidden.squeeze()  # (batch_size, emb_dim)
      
        # Unpack packed sequence
        w, _ = pad_packed_sequence(w, batch_first=True)

        row_indices = torch.arange(0, batch_size).long()
        col_indices = text_length - 1
        last_hidden = w[row_indices, col_indices, :]
 
        mean_tag = emb_tag.mean(dim=1)

        concatenates = torch.cat([last_hidden, mean_tag], dim=1)  # (batch_size, 2*emb_dim)
        out = self.linear(concatenates)  # (batch_size, 1)

        return torch.sigmoid(out)

class Arch3(nn.Module):
    def __init__(self, emb_num, emb_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(emb_num, emb_dim)
        self.lstm = nn.LSTM(emb_dim * 2, emb_dim, 
                            batch_first=True)
        self.linear = nn.Linear(emb_dim, 1)
    
    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).

        :param fine_tune: Fine-tune?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
    
    def forward(self, text, tag, text_length):
        emb_text = self.embedding(text)  # (batch_size, seq_len, emb_dim)
        emb_tag = self.embedding(tag)  # (batch_size, topic_len, emb_dim)
        mean_tag = emb_tag.mean(dim=1) #(batch_size, emb_dim)

        seqs = []
        for i in range(emb_text.size()[1]):
            concat = torch.cat([emb_text[:, i, :], mean_tag], dim=1)  # (batch_size, emb_dim*2)
            seqs.append(concat)

        seqs = torch.stack(seqs)  # (seq_len, batch_size, emb_dim*2)
        seqs = seqs.transpose(0, 1)  # (batch_size, seq_len, emb_dim*2)

        _, (last_hidden, _) = self.lstm(seqs)
        last_hidden = last_hidden.squeeze()  # (batch_size, emb_dim)

        out = self.linear(last_hidden)  # (batch_size, 1)
        return torch.sigmoid(out)