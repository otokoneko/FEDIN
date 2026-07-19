# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import math

import numpy as np
import torch
from torch import nn
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, MultiHeadTargetAttention
from torch.nn import MultiheadAttention


class DSIN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DSIN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=32,
                 max_sequence_len=430,
                 num_heads=1,
                 batch_norm=True,
                 target_field=[("item_id", "cate_id")],
                 sequence_field=[("click_history", "cate_history")],
                 session_prefix='s{}_',
                 session_num=5,
                 seq_pooling_type="mean",  # ["mean", "sum"]
                 use_causal_mask=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 net_dropout=0.0,
                 attention_dropout=0.0,
                 dnn_hidden_units=[64, 32],
                 dnn_hidden_activations='sigmoid',
                 **kwargs):
        super(DSIN, self).__init__(feature_map,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.max_sequence_len = max_sequence_len
        self.seq_pooling_type = seq_pooling_type
        self.use_causal_mask = use_causal_mask
        self.num_heads = num_heads
        self.sessions = [session_prefix.format(i) for i in range(session_num)]
        if not isinstance(target_field, list):
            target_field = [target_field]
        self.target_field = target_field
        if not isinstance(sequence_field, list):
            sequence_field = [sequence_field]
        self.sequence_field = sequence_field
        assert len(self.target_field) == len(self.sequence_field), \
            "len(target_field) != len(sequence_field)"

        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.position_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.target_attention_layers = nn.ModuleList()
        extra_dim = 0
        for target_field in self.target_field:
            input_dim = embedding_dim * len(target_field) if isinstance(target_field, tuple) else embedding_dim
            extra_dim += 2 * input_dim
            extra_dim -= session_num * input_dim
            self.position_layers.append(nn.Embedding(self.max_sequence_len + 1, input_dim))
            self.attention_layers.append(
                DSINAttention(input_dim=input_dim, dropout_rate=attention_dropout, num_heads=num_heads))
            self.lstm_layers.append(
                nn.LSTM(input_dim, input_dim, bidirectional=True, proj_size=input_dim // 2))
            self.target_attention_layers.append(
                MultiHeadTargetAttention(input_dim=input_dim, dropout_rate=attention_dropout, use_scale=True,
                                         use_qkvo=False))
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim() + extra_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_hidden_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        for target_field, sequence_field, attention_layer, position_layer, lstm_layer, target_attention_layer in zip(
                self.target_field,
                self.sequence_field,
                self.attention_layers,
                self.position_layers,
                self.lstm_layers, self.target_attention_layers):
            target_emb = self.get_embedding(target_field, feature_emb_dict)
            sequence_emb = self.get_session_embedding(sequence_field, feature_emb_dict)

            seq_field = list(flatten([sequence_field]))[0]
            padding_mask, attn_mask = self.get_mask(self.get_session_x(X, seq_field))

            transformer_out = attention_layer(sequence_emb, attn_mask)
            transformer_out = self.sequence_pooling(transformer_out, padding_mask)
            lstm_out, _ = lstm_layer(transformer_out)

            transformer_interest = target_attention_layer(target_emb, transformer_out)
            lstm_interest = target_attention_layer(target_emb, lstm_out)

            for sequence, target, field_emb_1, field_emb_2, tar_emb in zip(list(flatten([sequence_field])),
                                                                           list(flatten([target_field])),
                                                                           transformer_interest.split(
                                                                               self.embedding_dim, dim=-1),
                                                                           lstm_interest.split(self.embedding_dim,
                                                                                               dim=-1),
                                                                           target_emb.split(self.embedding_dim,
                                                                                            dim=-1)):
                for s in self.sessions:
                    feature_emb_dict.pop(s + sequence)
                feature_emb_dict[self.sessions[0] + sequence] = torch.cat([field_emb_1, field_emb_2], dim=-1)
                feature_emb_dict[target] = tar_emb

        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def get_session_embedding(self, field, feature_emb_dict):
        emb_list = []
        for s in self.sessions:
            if type(field) == tuple:
                session_field = tuple(s + f for f in field)
            else:
                session_field = s + field
            emb_list.append(self.get_embedding(session_field, feature_emb_dict).unsqueeze(1))
        return torch.cat(emb_list, dim=1)

    def get_session_x(self, X, field):
        x = []
        for s in self.sessions:
            if type(field) == tuple:
                session_field = (s + f for f in field)
            else:
                session_field = s + field
            x.append(X[session_field].unsqueeze(1))
        return torch.cat(x, dim=1)

    def get_mask(self, x):
        """ padding_mask: 1 for masked positions
            attn_mask: 1 for masked positions in nn.MultiheadAttention
        """
        B, M, T = tuple(x.size())
        x = x.view(-1, T)
        padding_mask = (x == 0)
        seq_len = padding_mask.size(1)
        attn_mask = padding_mask.unsqueeze(1).repeat(1, seq_len * self.num_heads, 1).view(-1, seq_len, seq_len)
        diag_zeros = (1 - torch.eye(seq_len, device=x.device)).bool().unsqueeze(0).expand_as(attn_mask)
        attn_mask = attn_mask & diag_zeros
        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1).bool() \
                .unsqueeze(0).expand_as(attn_mask)
            attn_mask = attn_mask | causal_mask
        return padding_mask.view(B, M, T), attn_mask

    def sequence_pooling(self, transformer_out, mask):
        mask = (1 - mask.float()).unsqueeze(-1)  # 0 for masked positions
        if self.seq_pooling_type == "mean":
            return (transformer_out * mask).sum(dim=2) / (mask.sum(dim=2) + 1.e-12)
        elif self.seq_pooling_type == "sum":
            return (transformer_out * mask).sum(dim=2)
        else:
            raise ValueError("seq_pooling_type={} not supported.".format(self.seq_pooling_type))


class DSINAttention(nn.Module):
    def __init__(self, input_dim=64, dropout_rate=0.0, num_heads=1):
        super().__init__()
        self.attention = MultiheadAttention(input_dim, dropout=dropout_rate, num_heads=num_heads, batch_first=True)

    def forward(self, x, attn_mask=None):
        B, M, T, H = tuple(x.size())
        x = x.view(B * M, T, H)
        attn, _ = self.attention(x, x, x, attn_mask=attn_mask)
        return attn.view(B, M, T, H)
