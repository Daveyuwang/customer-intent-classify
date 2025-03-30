"""
Combined BERT+CNN+LSTM model with attention for intent classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import MODEL_NAME, MAX_LEN, N_CLASS, DEVICE

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
voc_size = len(tokenizer.vocab)

################################################################################
# Attention modules
################################################################################
class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class ScaleDotProductAttention(nn.Module):
    """
    Scale dot product attention mechanism.
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # q, k, v: (batch_size, head, length, d_tensor)
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        out = score @ v
        return out, score

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    """
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # Split into heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        # tensor shape: (batch_size, length, d_model)
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        # tensor shape: (batch_size, head, length, d_tensor)
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.
    """
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """
    Encoder layer with self-attention and feed-forward networks.
    """
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=None)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

################################################################################
# Sub-blocks for the combined model
################################################################################

class BertBlock(nn.Module):
    """
    BERT sub-block that outputs a 100-dim feature vector.
    """
    def __init__(self):
        super().__init__()
        # A BERT model that outputs a 100-dim logit layer
        self.model_block = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=100
        ).to(DEVICE)

    def forward(self, text_b):
        x = self.model_block(text_b)
        return x.logits  # shape: (batch_size, 100)

class TextCNNBlock(nn.Module):
    """
    CNN sub-block that outputs a 100-dim feature vector.
    """
    def __init__(self):
        super().__init__()
        emb_dim = 100
        kernels = [3, 4, 5]
        kernel_number = [150, 150, 150]
        self.embd = nn.Embedding(voc_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(MAX_LEN, kn, k, padding=k) 
            for (k, kn) in zip(kernels, kernel_number)
        ])
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(sum(kernel_number), 100)  # output is 100-dim

    def forward(self, x):
        x = self.embd(x)  # (batch_size, 25, 100)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)  # (batch_size, 450)
        x = self.dropout(x)
        x = self.out(x)      # (batch_size, 100)
        return x

class LSTMBlock(nn.Module):
    """
    LSTM sub-block that outputs a 100-dim feature vector.
    """
    def __init__(self):
        super().__init__()
        emb_dim = 100
        self.embd = nn.Embedding(voc_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, 50, batch_first=True)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(50 * MAX_LEN, 100)  # 50*25=1250 -> 100

    def forward(self, x):
        x = self.embd(x)            # (batch_size, 25, 100)
        x, _ = self.lstm(x)         # (batch_size, 25, 50)
        x = self.flatten(x)         # (batch_size, 1250)
        x = self.out(x)             # (batch_size, 100)
        return x

################################################################################
# Combined model (BERT + CNN + LSTM + attention)
################################################################################
class BERT_CNN_LSTM(nn.Module):
    """
    Combined model that leverages BERT, CNN, and LSTM with attention.
    
    The model extracts features from each sub-model and uses attention
    to combine them effectively for intent classification.
    """
    def __init__(self):
        super(BERT_CNN_LSTM, self).__init__()
        self.bert = BertBlock()
        self.lstm = LSTMBlock()
        self.cnn = TextCNNBlock()

        # Attention layer over 100-dim, with shape (batch, 3, 100)
        self.att = EncoderLayer(d_model=100, ffn_hidden=200, n_head=2, drop_prob=0.1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(300, 100)  # 3 sub-blocks each = 100 → 300
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, N_CLASS)

    def forward(self, input_ids):
        """
        Forward pass through the combined model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Logits for each class of shape (batch_size, n_class)
        """
        # Sub-blocks each return (batch_size, 100)
        bert_out = self.bert(input_ids)
        lstm_out = self.lstm(input_ids)
        cnn_out = self.cnn(input_ids)

        # Stack to get shape = (batch_size, 3, 100)
        x = torch.stack((bert_out, lstm_out, cnn_out), dim=1)
        x = self.att(x)       # Shape remains (batch_size, 3, 100)
        x = self.flatten(x)   # (batch_size, 300)

        x = self.fc1(x)       # (batch_size, 100)
        x = self.dropout1(x)
        x = self.fc2(x)       # (batch_size, 11)
        return x