#   ______                                           __                 
#  /      \                                         /  |                
# /$$$$$$  | __   __   __   ______   _______        $$ |       __    __ 
# $$ |  $$ |/  | /  | /  | /      \ /       \       $$ |      /  |  /  |
# $$ |  $$ |$$ | $$ | $$ |/$$$$$$  |$$$$$$$  |      $$ |      $$ |  $$ |
# $$ |  $$ |$$ | $$ | $$ |$$    $$ |$$ |  $$ |      $$ |      $$ |  $$ |
# $$ \__$$ |$$ \_$$ \_$$ |$$$$$$$$/ $$ |  $$ |      $$ |_____ $$ \__$$ |
# $$    $$/ $$   $$   $$/ $$       |$$ |  $$ |      $$       |$$    $$/ 
#  $$$$$$/   $$$$$/$$$$/   $$$$$$$/ $$/   $$/       $$$$$$$$/  $$$$$$/ 
#
# File: decoder.py
# Author: Owen Lu
# Date: 2021/8/8
# Email: jiangxiluning@gmail.com
# Description:
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
from numpy import random
from easydict import EasyDict
import einops

from ..data_module import vocabs

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    # seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self, config: EasyDict):
        super(Decoder, self).__init__()

        self.embedding_dim = config.model.embedding_dim
        self.hidden_dim = config.model.hidden_dim
        self.pointer_gen = config.model.pointer_gen

        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(len(vocabs.key), self.embedding_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(self.hidden_dim * 2 + self.embedding_dim, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.p_copy_linear = nn.Linear(self.hidden_dim * 4 + self.embedding_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.out2 = nn.Linear(self.hidden_dim, len(vocabs.key))
        init_linear_wt(self.out2)

    def forward(self,
                y_t_1,
                s_t_1_forward,
                s_t_1_backward,
                class_emb,
                encoder_outputs_forward,
                encoder_outputs_backward,
                encoder_inputs,
                enc_padding_mask,
                c_t_1_forward,
                c_t_1_backward,
                coverage_forward,
                coverage_backward,
                step):
        """

        Args:
            y_t_1: B * num_entities
            s_t_1_forward: (hidden, context) hidden: B * num_entities * hidden_dim,
                           context: B * num_entities * hidden_dim
            s_t_1_backward: (hidden, context) hidden: B * num_entities * hidden_dim,
                           context: B * num_entities * hidden_dim
            class_emb: B * num_entities * class_embedding_dim
            encoder_outputs_forward: B * max_len_encoder
            encoder_outputs_backward: B * max_len_encoder
            encoder_inputs: B * max_len_encoder
            enc_padding_mask:
            c_t_1_forward: B * num_entities * embedding_dim
            c_t_1_backward:  B * num_entities * embedding_dim
            coverage_forward: B * num_entities * max_len_encoder
            coverage_backward:  B * num_entities * max_len_encoder
            step:

        Returns:

        """

        if not self.training and step == 0:
            h_decoder_forward, c_decoder_forward = s_t_1_forward
            s_t_hat_forward = torch.cat((h_decoder_forward.view(-1, self.hidden_dim),
                                         c_decoder_forward.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t_forward, _, coverage_next_forward = self.attention_network(s_t_hat_forward,
                                                                           encoder_outputs_forward,
                                                                           encoder_feature,
                                                                           enc_padding_mask,
                                                                           coverage_forward)
            coverage_forward = coverage_next_forward

            h_decoder_backward, c_decoder_backward = s_t_1_backward
            s_t_hat_backward = torch.cat((h_decoder_backward.view(-1, self.hidden_dim),
                                          c_decoder_backward.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t_backward, _, coverage_next_backward = self.attention_network(s_t_hat_backward,
                                                                             encoder_outputs_backward,
                                                                             encoder_feature,
                                                                             enc_padding_mask,
                                                                             coverage_forward)
            coverage_backward = coverage_next_backward

        y_t_1_embd = self.embedding(y_t_1)
        x_forward = self.x_context(torch.cat((c_t_1_forward, y_t_1_embd), 1))
        x_backward = self.x_context(torch.cat((c_t_1_backward, y_t_1_embd), 1))

        _, s_t_forward = self.lstm(x_forward.unsqueeze(1), s_t_1_forward)
        _, s_t_backward = self.lstm(x_backward.unsqueeze(1), s_t_1_backward)

        h_decoder_forward, c_decoder_forward = s_t_forward
        s_t_hat_forward = torch.cat((h_decoder_forward.view(-1, self.hidden_dim),
                                     c_decoder_forward.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t_forward, attn_dist_forward, coverage_next_forward = self.attention_network(s_t_hat_forward,
                                                                                       encoder_outputs_forward,
                                                                                       encoder_feature,
                                                                                       enc_padding_mask,
                                                                                       coverage_forward)

        h_decoder_backward, c_decoder_backward = s_t_backward
        s_t_hat_backward = torch.cat((h_decoder_backward.view(-1, self.hidden_dim),
                                      c_decoder_backward.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t_backward, attn_dist_backward, coverage_next_backward = self.attention_network(s_t_hat_backward,
                                                                                          encoder_outputs_backward,
                                                                                          encoder_feature,
                                                                                          enc_padding_mask,
                                                                                          coverage_backward)

        if self.training or step > 0:
            coverage_forward = coverage_next_forward
            coverage_backward = coverage_next_backward

        F_aggregated = torch.cat((c_t_forward, c_t_backward))
        s_t_aggregated = torch.cat((s_t_hat_forward, s_t_hat_backward))

        p_copy_input = torch.cat((F_aggregated, s_t_aggregated, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        p_copy = self.p_gen_linear(p_copy_input)
        p_copy = F.sigmoid(p_copy)

        output = torch.cat((F_aggregated, s_t_aggregated), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim
        output = self.out2(output)  # B x vocab_size

        p_pred = torch.softmax(output, dim=1)

        index = torch.where((encoder_inputs == y_t_1))

        selected_attn_forward = attn_dist_forward[index]
        selected_attn_backward = attn_dist_backward[index]

        selected_attn = (selected_attn_forward + selected_attn_backward) / 2

        p_output = p_copy * selected_attn + (1 - p_copy) * p_pred[y_t_1]

        return p_output, s_t_forward, s_t_backward, c_t_forward, c_t_backward, coverage_forward, coverage_backward
