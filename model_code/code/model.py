import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.autograd as autograd

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class LALoss(nn.Module):
#     ## logit adjustement loss function
#
#     def __init__(self, cls_num_list, tau=1.0):
#         super(LALoss, self).__init__()
#         base_probs = cls_num_list / cls_num_list.sum()
#         scaled_class_weights = tau * torch.log(base_probs + 1e-12)
#         scaled_class_weights = scaled_class_weights.reshape(1, -1)  # [1,classnum]
#         self.scaled_class_weights = scaled_class_weights.float().cuda()
#
#     def forward(self, x, target):
#         x += self.scaled_class_weights
#         return F.cross_entropy(x, target)

class LALossBinary(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super(LALossBinary, self).__init__()
        base_probs = cls_num_list / cls_num_list.sum()  
        scaled_class_weights = tau * torch.log(base_probs + 1e-12)  
        self.pos_weight = scaled_class_weights[1].float().cuda() 
        self.neg_weight = scaled_class_weights[0].float().cuda()  

    def forward(self, logits, target):
        logits = logits.squeeze(-1) 
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=self.pos_weight
        )
        return bce_loss
class ConfigTrans(object):
    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = 128
        self.pad_size = 4
        self.embed = 768
        self.dim_model = 768
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 8
        self.num_encoder = 6


confi = ConfigTrans()


class Multi_Head_Attention(nn.Module):
    '''
    params: dim_model-->hidden dim      num_head
    '''
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head

        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)

        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)


    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)  
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)  
        out = self.dropout(out)
        out = out + x 
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder(confi.dim_model, confi.num_head, confi.hidden, confi.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(confi.num_encoder)])

        self.fc1 = nn.Linear(confi.pad_size * confi.dim_model, confi.num_classes)

    def forward(self, x): # [4, 3, 768]->[4, 128]
        # out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(x)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,  # 768
                      out_channels=out_channels,  # 128
                      kernel_size=fs)  # 1, 2, 3
            for fs in filter_sizes
        ])


        self.init_params()
    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            # nn.init.xavier_uniform_：
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):  # [4, 768, 3] -> # [4, 128, 1] / [4, 128, 2] / [4, 128, 3]
        return [F.relu(conv(x)) for conv in self.convs]


class Linear(nn.Module):
    def __init__(self, in_features, out_features):  # in_features=2, out_features=128
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)

        nn.init.constant_(self.linear.bias, 0)
        # nn.init.constant_：
    def forward(self, x):
        x = self.linear(x)
        return x

class TextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):  # embedding_dim = 768;   n_filters = 128;   filter_sizes = [1, 2, 3];   output_dim = 128;
        super().__init__()
        self.convs = Conv1d(embedding_dim, n_filters,
                            filter_sizes)  # embedding_dim = 768;   n_filters = 128;   filter_sizes = [1, 2, 3];
        self.fc = Linear(len(filter_sizes) * n_filters,
                         output_dim)  # len(filter_sizes) * n_filters = 2*1=2, output_dim = 128
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [4, 3, 768]
        embedded = x.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]
        # [4, 3, 768] -> # [4, 768, 3]
        conved = self.convs(embedded)  # [4, 768, 3] -> # [4, 128, 1] / [4, 128, 2] / [4, 128, 3]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        # conv [batch_size, n_filters, sent_len - filter_size + 1]。
        # output  [batch_size, n_filters]。 [4, 128, 1] -> [4, 128, 1]///[4, 128, 2] -> [4, 128, 1]///[4, 128, 3] -> [4, 128, 1]
        # squeeze(2)：-> [batch_size, n_filters].[4, 128, 1] -> [4, 128]
        cat = self.dropout(torch.cat(pooled, dim=1))  # [B, n_filters * len(filter_sizes)] [4, 128*3]
        # [[4, 128], [4, 128], [4, 128]] -> [4, 3*128]
        return self.fc(cat)
        # [4, 128*3] -> [4, 128]

class CNNClassificationSeq_Cnn_Transformer(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.d_size = self.args.d_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

        self.dense = nn.Linear(self.d_size, config.hidden_size)

        # CNN
        self.window_size = self.args.cnn_size
        self.filter_size = []
        for i in range(self.args.filter_size):  # filter_size：default=3
            i = i + 1
            self.filter_size.append(i)  # [1, 2]
        self.cnn = TextCNN(config.hidden_size, self.window_size, self.filter_size, self.d_size, 0.2)
        # Transformer
        self.transformer = Transformer()
        # self.linear_mlp = nn.Linear(config.hidden_size, self.d_size)
        self.gate_fc = nn.Linear(2 * self.d_size, self.d_size)  # gate
    def forward(self, features, **kwargs): # [4, 3*768]
        # features: [batch_size, L * D]
        x = torch.unsqueeze(features, dim=1)  # [B, L*D] -> [B, 1, L*D]  # [4, 3*768] -> # [4, 1, 3*768]
        x = x.reshape(x.shape[0], -1, 768)  # [B, L, D] -> [batch_size, L, D]  # [4, 3, 768]
        # CNN
        # [4, 3, 768]
        outputs = self.cnn(x)  # [B, L, D]->[B, D]  # [4, 3, 768] -> [4, 128]
        # Transformer
        transformer_features = self.transformer(x)  # [B, L, D] -> [B, dim_model]  # [4, 3, 768] -> [4, 128]
        # x = torch.cat((transformer_features, outputs), dim=-1)
        # [4, 128] + [4, 128] -> [4, 2*128]
        combined = torch.cat((transformer_features, outputs), dim=-1)  # [4, 128] + [4, 128] -> [4, 256]
        gate = torch.sigmoid(self.gate_fc(combined))
        fused_features = gate * transformer_features + (1 - gate) * outputs


        x = self.dropout(fused_features)
        x = self.dense(x)
        # [4, 2*128] -> [4, 768]
        # ------------- cnn ----------------------
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        # [4, 768] -> [4, 1]
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.linear = nn.Linear(3,
                                1)
        self.cnnclassifier = CNNClassificationSeq_Cnn_Transformer(config,
                                                  self.args)
        self.cnnclassifier_Textcnn_And_Transformer = CNNClassificationSeq_Cnn_Transformer(config,
                                                  self.args)
    def forward(self, seq_ids=None, input_ids=None, labels=None):
        # [4, 3, 400]
        batch_size = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        token_len = seq_ids.shape[-1]
        # print("seq_len = ", seq_len)
        seq_inputs = seq_ids.reshape(-1, token_len)  # [4, 3, 400] -> [4*3, 400]

        seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]  # [4*3, 400] -> [4*3, 400, 768]

        seq_embeds = seq_embeds[:, 0, :]  # [4*3, 400, 768] -> [4*3, 768]
        outputs_seq = seq_embeds.reshape(batch_size, -1)  # [4*3, 768] -> [4, 3*768]
        logits_path = self.cnnclassifier_Textcnn_And_Transformer(outputs_seq)

        prob_path = torch.sigmoid(logits_path)
        prob = prob_path

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
        #     # class_prior = torch.tensor([0.9, 0.1])
        #     # tau = 1.0
        #     # log_prior = torch.log(class_prior + 1e-10).to(labels.device)
        #     # adjusted_logits = prob - tau * log_prior
        #     # adjusted_prob = torch.sigmoid(adjusted_logits)
        #     # loss = (
        #     #         torch.log(adjusted_prob[:, 0] + 1e-10) * labels
        #     #         + torch.log(1 - adjusted_prob[:, 0] + 1e-10) * (1 - labels)
        #     # )
        #     # loss = -loss.mean()  
        #     # labels=labels.long()
        #     # class_priors = torch.tensor([0.9, 0.1])
        #     # adjusted_logits = prob + torch.log(class_priors + 1e-10).to(labels.device)
        #     # loss = F.cross_entropy(adjusted_logits, labels)
        #     # loss = loss.mean()
            return loss, prob
        else:
            return prob


if __name__ == '__main__':
    main()
