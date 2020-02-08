# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


# The code is based on repository: https://github.com/yangheng95/LCF-ABSA


import torch
import torch.nn as nn
import copy
import numpy as np

from transformers.modeling_albert import AlbertAttention


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = AlbertAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zeros = np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                         dtype=np.float32)
        zeros_type = torch.float32
        zero_tensor = torch.tensor(zeros, dtype=zeros_type).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class ModifiedAlbertPooler(nn.Module):
    '''
    Normally, albert by default only pools with
    a dense layer, seen in `forward(...)`.

    The actual implementation can be found in 
    transformers/modeling_albert.py in AlbertModel
    under `pooled_output`

    We add a Tanh activation and pool the model
    by hidden state of the first token
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LCF_ALBERT(nn.Module):
    def __init__(self, albert, opt):
        super(LCF_ALBERT, self).__init__()

        self.albert = albert
        # self.albert_local = copy.deepcopy(albert)  # Uncomment the line to use dual Albert
        self.albert_local = albert   # Default to use single Albert and reduce memory requirements

        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.attention = SelfAttention(albert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.albert_pooler = ModifiedAlbertPooler(albert.config)

        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((
                                           text_local_indices.size(0),
                                           self.opt.max_seq_len,
                                           self.opt.bert_dim),
                                          dtype=np.float32)

        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except Exception as e:
                print(e)
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except Exception as e:
                print(e)
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i])-1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index)+asp_len/2
                                        - self.opt.SRD)/np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_indices = inputs[0]
        albert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        albert_out, _ = self.albert(text_indices, albert_segments_ids)
        albert_out = self.dropout(albert_out)

        albert_local_out, _ = self.albert_local(text_local_indices)
        albert_local_out = self.dropout(albert_local_out)

        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(
                text_local_indices,
                aspect_indices
            )

            albert_local_out = torch.mul(
                albert_local_out,
                masked_local_text_vec
            )

        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(
                text_local_indices,
                aspect_indices
            )

            albert_local_out = torch.mul(
                albert_local_out,
                weighted_text_local_features
            )

        out_cat = torch.cat((albert_local_out, albert_out), dim=-1)
        mean_pool = self.linear_double(out_cat)
        self_attention_out = self.attention(mean_pool)
        pooled_out = self.albert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)

        return dense_out
