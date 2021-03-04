import torch

import torch.nn as nn

import math

from torch.autograd import Variable
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):

        super(Embedding, self).__init__()

        self.lut = nn.Embedding(vocab, d_model)

        self.d_model = d_model

    def forward(self, x):

        return self.lut(x) * math.sqrt(self.d_model)

# embedding = nn.Embedding(10, 3)
# input = torch.LongTensor([[1,2,4,5], [4,3,2,9]])
# print(embedding(input))

d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))

emb = Embedding(d_model, vocab)
embr = emb(x)
x = embr
# print('embr:', embr)
# print(embr.shape)


# 位置编码器的作用
'''因为在transformer的编码器结构中并没有针对词汇位置信息的处理，因此需要embedding层后加入
位置编码器，将词汇位置不同可能会产生不同语义的信息加入到词嵌入张量中，以弥补位置信息的缺失'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        postion = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(postion * div_term)
        pe[:, 1::2] = torch.cos(postion * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

d_model = 512
dropout = 0.1
max_len = 60

pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
# print("pe_result:", pe_result)

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(25,15))

pe = PositionalEncoding(20, 0)

y = pe(Variable(torch.zeros(1, 100, 20)))

plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())

plt.legend(["dim %d"%p for p in [4,5,6,7]])
plt.show()


# 2.3 编码器部分实现
# 生成掩码张量的代码分析
def subsequent_mask(size):

    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(1 - subsequent_mask)

size = 5

sm = subsequent_mask(size)
print('sm:', sm)
print(sm.shape)

plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
plt.show()
print(sm.shape)


# 注意力机制
def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1) / math.sqrt(d_k))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


query = key = value = pe_result
attn, p_attn = attention(query, key, value)
print('attn:', attn)
print(attn.shape)
print('p_attn:', p_attn)
print(p_attn.shape)

query = key = value = pe_result

mask = Variable(torch.zeros(2, 4, 4))
attn, p_attn = attention(query, key, value, mask=mask)
print('attn:', attn)
print(attn.shape)
print('p_attn:', p_attn)
print(p_attn.shape)


# 多头注意力机制
'''这种结构设计能让每个注意力机制去优化每个词汇的不同部分，从而均衡同一种注意力机制可能产生的偏差，
让词义拥有更多元的表达，实验表明可以从而提升模型效果'''

import copy

def clones(module, N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadeAttention(nn.Module):

    def __init__(self, head, embedding_dim, dropout=0.1):

        super(MultiHeadeAttention, self).__init__()

        assert embedding_dim % head == 0

        self.d_k = embedding_dim // head

        self.head = head

        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:

            mask = mask.unsqueeze(0)

        batch_size = query.size(0)


        query, key, value = \
        [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
         for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.linears[-1](x)

head = 8

embedding_dim = 512

dropout = 0.2

query = value = key = pe_result

mask = Variable(torch.zeros(8, 4, 4))

mha = MultiHeadeAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
x = mha_result
# print(mha_result)


# 前馈连接层
'''在transformer中前馈连接层就是具有两层线性层的全连接网络
    考虑注意力机制可能对复杂过程的拟合程度不够，通过增加两层网络来增强模型的能力'''

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):

        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.w2(self.dropout(F.relu(self.w1(x))))

d_model = 512
d_ff = 64
dropout = 0.2

ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
# print(ff_result)
x = ff_result

# 规范化层
'''他是所有深层网络模型都需要的标准网络层，因为随着网络层数的增加，通过多层的计算后参数可能开始
    出现过大或过小的情况，这样可能会导致学习过程出现异常，模型可能收敛非常的慢，因此都会在一定层数后接
    规范化层进行数值的规范化，使其特征数值在合理范围内'''

class LayerNorm(nn.Module):

    def __init__(self, features, eps = 1e6):

        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

feature = d_model = 512
eps = 1e-6

ln = LayerNorm(feature, eps)
ln_result = ln(x)
# print(ln_result)
# print(ln_result.shape)


# 子连接层
'''输入到每个子层以及规范化层的过程中，还使用了残差连接（跳跃连接），因此我们把这一部分结构
    整体叫做子层连接（代表子层及其链接结构），在每个编码器层中，都有两个子层，这两个子层
    加上周围的链接结构就形成了两个子层连接结构'''

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):

        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))

size = 512
dropout = 0.2
head = 8
d_model = 512

x = pe_result
mask = Variable(torch.zeros(8, 4, 4))
self_attn = MultiHeadeAttention(head, d_model)
sublayer = lambda x: self_attn(x, x, x, mask)

sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
# print(sc_result)
# print(sc_result.shape)


# 编码器层
'''作为编码器的组层单元，每个编码器层完成一次对输入的特征提取过程，即编码过程'''
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):

        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)

        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

size = 512
head = 8
d_model = 512
d_ff = 64
x = pe_result
dropout = 0.2
self_attn = MultiHeadeAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(8, 4, 4))

el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)
# print(el_result)
# print(el_result.shape)


# 编码器
'''编码器用于对输入进行指定的特征提取过程，也称为编码，由N个编码器层堆叠而成'''

class Encoder(nn.Module):
    def __init__(self, layer, N):

        super(Encoder, self).__init__()

        self.layers = clones(layer, N)

        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


size = 512
head = 8
d_model = 512
d_ff = 64
c = copy.deepcopy
attn = MultiHeadeAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)

N = 8
mask = Variable(torch.zeros(8, 4, 4))

en = Encoder(layer, N)
en_result = en(x, mask)
# print(en_result)
# print(en_result.shape)


# 2.4解码器部分实现
# 解码器层作用
'''作为解码器的组层单元，每个解码器层根据给定的输入向目标方向进行特征提取，即解码过程'''

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):

        super(DecoderLayer, self).__init__()

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):

        m = memory

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        x = self.sublayer[1](x, lambda  x: self.self_attn(x, m, m, source_mask))

        return self.sublayer[2](x, self.feed_forward)

head = 8
size = 512
d_model = 512
d_ff = 64
dropout = 0.2
self_attn = src_attn = MultiHeadeAttention(head, d_model, dropout)

ff = PositionwiseFeedForward(d_model, d_ff, dropout)

x = pe_result
memory = en_result

mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
# print(dl_result)
# print(dl_result.shape)


# 解码器
'''解码器的作用根据编码器的结果以及上一次预测的结果，对下一次可能出现的'值'进行特征表示'''
class Decoder(nn.Module):
    def __init__(self, layer, N):

        super(Decoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):

        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

size = 512
d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
attn = MultiHeadeAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
N = 8


x = pe_result
memory = en_result
mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

de = Decoder(layer, N)
de_result = de(x, memory, source_mask, target_mask)
# print(de_result)
# print(de_result.shape)


# 2.5输出部分实现
'''线性层作用，通过对上一步线性变化得到指定纬度的输出，也就是转换维度的作用
    softmax层作用，使最后一维的向量中的数字缩放到0-1的概率值域内，并满足他们的和为1'''

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):

        super(Generator, self).__init__()

        self.project = nn.Linear(d_model, vocab_size)


    def forward(self, x):

        return F.log_softmax(self.project(x), dim=-1)


# 词嵌入维度
d_model = 512
# 词表大小
vocab_size = 1000

# 输入参数
x = de_result


gen = Generator(d_model, vocab_size)
gen_result = gen(x)
# print(gen_result)
# print(gen_result.shape)


# 2.6模型构建
'''编码器-解码器结构的代码实现'''
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):

        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):

        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):

        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):

        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
generator = gen

source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed_result = ed(source, target, source_mask, target_mask)
# print(ed_result)
# print(ed_result.shape)


# transformer模型构建
def make_model(source_vocab, target_vocab, N = 6, d_model= 512, d_ff = 2048, head = 8, dropout=0.1):

    c = copy.deepcopy

    attn = MultiHeadeAttention(head, d_model)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, source_vocab), c(position)),
        nn.Sequential(Embedding(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

source_vocab = 11
target_vocab = 11
N = 6

if __name__ == '__main__':
    res = make_model(source_vocab, target_vocab, N)
    print(res)










