import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import ModelInterface
from torch.cuda.amp import autocast
import math
from torchreid.losses import AngleSimpleLinearV2

def gen_A(num_classes, t, rho, adj_file):
    print(f"ACTUAL MATRIX PARAMS: {t}, {rho}")
    _adj = np.load(adj_file)
    # t = 0.1
    # rho = 0.2
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    if rho != 0.0:
        _adj = _adj * rho / (_adj.sum(0, keepdims=True) + 1e-6)
        _adj = _adj + np.identity(num_classes, np.int)
    return _adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, _input, adj):
        support = torch.matmul(_input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Image_GCNN(ModelInterface):
    def __init__(self, backbone, word_matrix, in_channel=300, adj_matrix=None, num_classes=80,
                 hidden_dim=1024, emb_dim=2048, **kwargs):
        super().__init__(**kwargs)
        print(f"ACTUAL DIMS: {hidden_dim}, {emb_dim}")
        self.backbone = backbone
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)
        self.gc1 = GraphConvolution(in_channel, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, emb_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.inp = nn.Parameter(torch.from_numpy(word_matrix).float())
        self.A = nn.Parameter(torch.from_numpy(adj_matrix).float())
        self.proj_embed = nn.Linear(self.backbone.num_features, self.num_classes * emb_dim, bias=False)
        self.proj_embed.weight = torch.nn.init.xavier_normal_(self.proj_embed.weight)

    def forward(self, image, return_embedings=False):
        with autocast(enabled=self.mix_precision):
            feature = self.backbone(image, return_featuremaps=True)
            in_size = feature.size()
            glob_features = feature.view((in_size[0], in_size[1], -1)).mean(dim=2)
            embedings = self.proj_embed(glob_features)
            embedings = embedings.reshape(image.size(0), self.num_classes, -1)

            adj = self.gen_adj(self.A).detach()
            x = self.gc1(self.inp, adj)
            x = self.relu(x)
            x = self.gc2(x, adj)

            if self.loss == 'am_binary':
                logits = F.cosine_similarity(embedings, x, dim=2)
                logits = logits.clamp(-1, 1)
            else:
                x = x.transpose(0, 1)
                assert self.loss in ['bce', 'asl']
                logits = torch.matmul(glob_features, x)

            if self.similarity_adjustment:
                logits = self.sym_adjust(logits, self.similarity_adjustment)

            if not self.training:
                return [logits]

            elif self.loss in ['asl', 'bce', 'am_binary']:
                if return_embedings:
                    return tuple([logits]), x, embedings
                else:
                    out_data = [logits]

            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))

            return tuple(out_data)

    @staticmethod
    def gen_adj(A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def get_config_optim(self, lrs):
        parameters = [
            {'params': self.proj_embed.named_parameters()},
            {'params': self.backbone.named_parameters()},
            {'params': self.gc1.named_parameters()},
            {'params': self.gc2.named_parameters()},
        ]
        if isinstance(lrs, list):
            assert len(lrs) == len(parameters)
            for lr, param_dict in zip(lrs, parameters):
                param_dict['lr'] = lr
        else:
            assert isinstance(lrs, float)
            for i, param_dict in enumerate(parameters):
                    param_dict['lr'] = lrs

        return parameters


def build_image_gcn(backbone, word_matrix_path, adj_file, num_classes=80, word_emb_size=300,
                    thau = 0.4, rho_gcn=0.25, pretrain=False, **kwargs):
    adj_matrix = gen_A(num_classes, thau, rho_gcn, adj_file)
    word_matrix = np.load(word_matrix_path)
    model = Image_GCNN(
        backbone=backbone,
        word_matrix=word_matrix,
        adj_matrix=adj_matrix,
        pretrain=pretrain,
        in_channel=word_emb_size,
        num_classes=num_classes,
        **kwargs
    )
    return model
