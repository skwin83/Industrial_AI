# H. Choi, borrowed a lot lines from "Yu-Hsiang Huang"

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from mylib.layers import CudaVariable, myEmbedding, myLinear, LayerNormalization

import nmt_const as Const

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def pos_encoding(n_position, dim_wemb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim_wemb) for j in range(dim_wemb)]
        if pos != 0 else np.zeros(dim_wemb) for pos in range(n_position)])

    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(pos_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    Bn, Tq = seq_q.size()
    Bn, Tk = seq_k.size()
    pad_attn_mask = seq_k.data.ne(Const.PAD).unsqueeze(1) # Bn 1 Tk
    pad_attn_mask = pad_attn_mask.expand(Bn, Tq, Tk) # Bn Tq Tk
    return pad_attn_mask.type(torch.cuda.FloatTensor)

def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.tril(np.ones(attn_shape)).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask).cuda()
    return subsequent_mask.type(torch.cuda.FloatTensor)

class PositionwiseFeedForwardOld(nn.Module):
    def __init__(self, d_hid, hid_dim, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_hid, hid_dim, 1) # position-wise
        self.w_2 = nn.Conv1d(hid_dim, d_hid, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + x)

class ScaledDotProductAttentionNew(nn.Module):
    def __init__(self, dim_dec, attn_dropout=0.1):
        super().__init__()
        self.temper = np.power(dim_dec, 0.5)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None): # HB T E
        # encY, encY, encY
        # decY, encY, encY
        #q = q / (torch.sum(q, 2, keepdim=True) + 1e-15)
        #k = k / (torch.sum(k, 2, keepdim=True) + 1e-15)

        attn = torch.bmm(q, k.transpose(1, 2))# / self.temper # HB Tq Tk
        attn = attn*attn_mask.type(torch.cuda.FloatTensor)
        attn = attn - torch.max(attn)

        attn = torch.exp(attn)*attn_mask.type(torch.cuda.FloatTensor)
        attn = attn / (torch.sum(attn, 2, keepdim=True) + 1e-15) 

        attn = self.dropout(attn) # HB Tq Tk
        output = torch.bmm(attn, v) # HB Tv E 

        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim, attn_dropout=0.1):
        super().__init__()
        self.temper = np.power(dim, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask=None): # HB T E
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper # HB Tq Tk
        attn.data.masked_fill_(attn_mask==0, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn) # HB Tq Tk
        output = torch.bmm(attn, v) # HB Tv E 

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, dim_dec, dk, dv, dropout=0.1):
        super().__init__()

        self.n_head, self.dk, self.dv = n_head, dk, dv

        self.w_qs = myLinear(dim_dec, n_head*dk)
        self.w_ks = myLinear(dim_dec, n_head*dk)
        self.w_vs = myLinear(dim_dec, n_head*dk)

        self.sdp_attn = ScaledDotProductAttentionNew(dim_dec)
        self.layer_norm = nn.LayerNorm(dim_dec)
        self.proj = myLinear(n_head*dv, dim_dec)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None): # (QK')V
        n_head, dk, dv = self.n_head, self.dk, self.dv

        Bn, Tq, _ = q.size()
        Bn, Tk, _ = k.size()
        Bn, Tv, _ = v.size()
        residual = q

        q = self.w_qs(q).view(Bn, Tq, n_head, dk)
        k = self.w_ks(k).view(Bn, Tk, n_head, dk)
        v = self.w_vs(v).view(Bn, Tv, n_head, dv)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, Tq, dk) # (H*Bn) Tq dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, Tk, dk) # (H*Bn) Tk dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, Tv, dv) # (H*Bn) Tv dv

        output = self.sdp_attn(q, k, v, attn_mask=attn_mask.repeat(n_head, 1,1))

        output = output.view(n_head, Bn, Tq, dv)
        output = output.permute(1, 2, 0, 3).contiguous().view(Bn, Tq, -1) # Bn Tq (H*dv)

        # project back to residual size
        output = self.dropout(self.proj(output))
        output = self.layer_norm(output + residual)

        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, hid_dim, dropout=0.1):
        super().__init__()
        self.w_1 = myLinear(d_hid, hid_dim) 
        self.w_2 = myLinear(hid_dim, d_hid)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_hid)

    def forward(self, x):
        output = self.relu(self.w_1(x))
        output = self.w_2(output)
        output = self.dropout(output)
        return self.layer_norm(output + x)

class EncoderLayer(nn.Module):
    def __init__(self, dim_dec, hid_dim, n_head, dk, dv, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_dec, dk, dv, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_dec, hid_dim, dropout=dropout)

    def forward(self, encX, self_mask=None):
        encY = self.self_attn(encX, encX, encX, attn_mask=self_mask)
        encY = self.pos_ffn(encY)
        return encY

class DecoderLayer(nn.Module):
    def __init__(self, dim_dec, hid_dim, n_head, dk, dv, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, dim_dec, dk, dv, dropout=dropout)
        self.enc_mh_attn = MultiHeadAttention(n_head, dim_dec, dk, dv, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_dec, hid_dim, dropout=dropout)

    def forward(self, decX, encY, self_mask=None, attn_mask=None):
        decY = self.self_attn(decX, decX, decX, attn_mask=self_mask)
        decY = self.enc_mh_attn(decY, encY, encY, attn_mask=attn_mask)
        decY = self.pos_ffn(decY)
        return decY

class Encoder(nn.Module):
    def __init__(self, src_words_n, max_length, n_layers=6, n_head=8, dk=64, dv=64,
            dim_wemb=512, dim_dec=512, hid_dim=1024, dropout=0.1):
        super().__init__()

        self.max_length = max_length
        self.src_emb = nn.Embedding(src_words_n, dim_wemb, padding_idx=Const.PAD)

        self.pos_enc = nn.Embedding(self.max_length, dim_wemb, padding_idx=Const.PAD)
        self.pos_enc.weight.data = pos_encoding(self.max_length, dim_wemb)
        self.pos_enc.weight.requires_grad = False

        self.layer_stack = nn.ModuleList([
            EncoderLayer(dim_dec, hid_dim, n_head, dk, dv, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):
        encX = self.src_emb(src_seq) # Word embedding look up
        encPos = self.pos_enc(torch.fmod(src_pos, self.max_length)) # Position Encoding
        encY = encX + encPos

        self_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            encY = enc_layer(encY, self_mask=self_mask)

        return encY

class Decoder(nn.Module):
    def __init__(self, trg_words_n, max_length, n_layers=6, n_head=8, dk=64, dv=64,
            dim_wemb=512, dim_dec=512, hid_dim=1024, dropout=0.1):
        super().__init__()

        self.max_length = max_length
        self.dec_emb = nn.Embedding(trg_words_n, dim_wemb, padding_idx=Const.PAD)

        self.pos_enc = nn.Embedding(self.max_length, dim_wemb, padding_idx=Const.PAD)
        self.pos_enc.weight.data = pos_encoding(self.max_length, dim_wemb)
        self.pos_enc.weight.requires_grad = False

        self.layer_stack = nn.ModuleList([
            DecoderLayer(dim_dec, hid_dim, n_head, dk, dv, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, trg_seq, trg_pos, src_seq, encY):
        decX = self.dec_emb(trg_seq) # Word embedding look up
        decPos = self.pos_enc(torch.fmod(trg_pos, self.max_length)) # Posision Encoding
        decY = decX + decPos

        mh_attn_pad_mask = get_attn_padding_mask(trg_seq, trg_seq)
        mh_attn_sub_mask = get_attn_subsequent_mask(trg_seq)
        self_mask = mh_attn_pad_mask * mh_attn_sub_mask
        enc_attn_mask = get_attn_padding_mask(trg_seq, src_seq)

        for dec_layer in self.layer_stack:
            decY = dec_layer(decY, encY, self_mask=self_mask, attn_mask=enc_attn_mask)

        return decY

class AttAllNMT(nn.Module):
    def __init__(self, args=None):
        super().__init__()
      
        src_words_n, trg_words_n = args.src_words_n, args.trg_words_n
        dim_wemb, dim_dec, dropout  = args.dim_wemb, args.dim_dec, args.dropout_p
        max_length = args.max_length + 2

        # for AttALL
        hid_dim, n_layers, n_head = args.attall_hid_dim, args.attall_n_layers, args.attall_n_head
        dk, dv = args.attall_dk, args.attall_dv

        self.encoder = Encoder(src_words_n, max_length, n_layers=n_layers, n_head=n_head,
                        dim_wemb=dim_wemb, dim_dec=dim_dec, hid_dim=hid_dim, dropout=dropout)
        self.decoder = Decoder(trg_words_n, max_length, n_layers=n_layers, n_head=n_head,
                        dim_wemb=dim_wemb, dim_dec=dim_dec, hid_dim=hid_dim, dropout=dropout)

        if os.path.exists(args.lm1_file):
            lm1 = torch.load(args.lm1_file)
            self.encoder.src_emb.weight = nn.Parameter(lm1.src_emb.weight, requires_grad=False)
        if os.path.exists(args.lm2_file):
            lm2 = torch.load(args.lm2_file)
            self.decoder.dec_emb.weight = nn.Parameter(lm2.src_emb.weight, requires_grad=False)

        self.trg_word_proj = myLinear(dim_dec, trg_words_n, bias=False)

        assert dim_dec == dim_wemb, \
            'To use the residual connections, the dimensions of outputs shall be the same.'

        self.trg_word_proj.weight = self.decoder.dec_emb.weight # Share the weight 
        self.prob_proj = nn.LogSoftmax(dim=1)

        self.criterion = nn.CrossEntropyLoss(size_average=False, reduce=False) # logsoft+CE

    def forward(self, x_data, x_mask, y_data, y_mask, y_lens):
        x_data = CudaVariable(torch.LongTensor(x_data)) # T B
        x_mask = CudaVariable(torch.LongTensor(x_mask)) # T B
        y_data = CudaVariable(torch.LongTensor(y_data)) # T B
        y_mask = CudaVariable(torch.LongTensor(y_mask)) # T B

        x_data = x_data.transpose(0, 1) # B T
        x_mask = x_mask.transpose(0, 1) # B T
        y_data = y_data.transpose(0, 1) # B T
        y_mask = y_mask.transpose(0, 1) # B T

        ym = (y_data.data[:,1:].ne(Const.PAD)).type(torch.cuda.FloatTensor) 
        xm = (x_data.data.ne(Const.PAD)).type(torch.cuda.FloatTensor)

        y_target = y_data[:,1:]
        y_data_in = y_data[:,:-1]
        y_mask_in = y_mask[:,:-1]
        #for i in range(len(y_lens)):
        #    if y_lens[i]+1 < y_data_in.size(1):
        #        y_data_in[i, y_lens[i]+1] = Const.PAD  # +1 for BOS/EOS
        #        y_mask_in[i, y_lens[i]+1] = 0          # +1 for BOS/EOS

        encY = self.encoder(x_data, x_mask) * xm.unsqueeze(2)
        decY = self.decoder(y_data_in, y_mask_in, x_data, encY) # B T E
        logit = self.trg_word_proj(decY)
        logit = logit.view(-1, logit.size(2)) # BT E

        loss = self.criterion(logit, y_target.contiguous().view(-1)) 
        loss = torch.sum(loss * ym.contiguous().view(-1))/x_data.size(0)

        return loss#, y_hat
