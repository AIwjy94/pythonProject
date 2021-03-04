import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchtext
from torchtext.data.utils import get_tokenizer
from pyitcast.transformer import TransformerModel


TEXT = torchtext.data.Field(tokenize=get_tokenizer('basic_english'),
                            init_token='<sos>', eos_token='<eos>', lower='True')

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)

TEXT.build_vocab(train_txt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data, bsz):

    data = TEXT.numericalize([data.examples[0].text])

    nbatch = data.size(0) // bsz

    data = data.narrow(0, 0, nbatch * bsz)

    data = data.view(bsz, -1).t().contiguous()

    return data.to(device)

batch_size = 30
eval_batch_size = 10

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)



# 批次化过程的第二个函数get_batch代码分析

bptt = 35

def get_batch(source, i):

    seq_len = min(bptt, len(source) - 1 - i)

    data = source[i:i+seq_len]

    target = source[i+1:i+1+seq_len].view(-1)

    return data, target

source = test_data
i = 1
data, target = get_batch(source, i)
print('data ====', data)
print('target ====', target)


# 第四步构建训练和评估函数
ntokens = len(TEXT.vocab.stoi)

emsize = 200

nhid = 200

nlayers = 2

nhead = 2

dropout = 0.2

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()

lr = 0.5

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# 导入时间工具包
import time

def train():
    """训练函数"""
    # 模型开启训练模式
    model.train()
    # 定义初始损失为0
    total_loss = 0.
    # 获得当前时间
    start_time = time.time()
    # 开始遍历批次数据
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # 通过get_batch获得源数据和目标数据
        data, targets = get_batch(train_data, i)
        # 设置优化器初始梯度为0梯度
        optimizer.zero_grad()
        # 将数据装入model得到输出
        output = model(data)
        # 将输出和目标数据传入损失函数对象
        loss = criterion(output.view(-1, ntokens), targets)
        # 损失进行反向传播以获得总的损失
        loss.backward()
        # 使用nn自带的clip_grad_norm_方法进行梯度规范化, 防止出现梯度消失或爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # 模型参数进行更新
        optimizer.step()
        # 将每层的损失相加获得总的损失
        total_loss += loss.item()
        # 日志打印间隔定为200
        log_interval = 200
        # 如果batch是200的倍数且大于0，则打印相关日志
        if batch % log_interval == 0 and batch > 0:
            # 平均损失为总损失除以log_interval
            cur_loss = total_loss / log_interval
            # 需要的时间为当前时间减去开始时间
            elapsed = time.time() - start_time
            # 打印轮数, 当前批次和总批次, 当前学习率, 训练速度(每豪秒处理多少批次),
            # 平均损失, 以及困惑度, 困惑度是衡量语言模型的重要指标, 它的计算方法就是
            # 对交叉熵平均损失取自然对数的底数.
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))

            # 每个批次结束后, 总损失归0
            total_loss = 0
            # 开始时间取当前时间
            start_time = time.time()


def evaluate(eval_model, data_source):

    eval_model.eval()

    total_loss = 0


    with torch.no_grad():

        for i in range(0, data_source.size(0) - 1, bptt):

            data, targets = get_batch(data_source, i)

            output = eval_model(data)

            output_flat = output.view(-1, ntokens)

            total_loss += criterion(output_flat, targets).item()

            cur_loss = total_loss / ((data_source.size(0) - 1) / bptt)

        return cur_loss


# 首先初始化最佳验证损失，初始值为无穷大
import copy
best_val_loss = float("inf")

# 定义训练轮数
epochs = 3

# 定义最佳模型变量, 初始值为None
best_model = None

# 使用for循环遍历轮数
for epoch in range(1, epochs + 1):
    # 首先获得轮数开始时间
    epoch_start_time = time.time()
    # 调用训练函数
    train()
    # 该轮训练后我们的模型参数已经发生了变化
    # 将模型和评估数据传入到评估函数中
    val_loss = evaluate(model, val_data)
    # 之后打印每轮的评估日志，分别有轮数，耗时，验证损失以及验证困惑度
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    # 我们将比较哪一轮损失最小，赋值给best_val_loss，
    # 并取该损失下的模型为best_model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 使用深拷贝，拷贝最优模型
        best_model = copy.deepcopy(model)
    # 每轮都会对优化方法的学习率做调整
    scheduler.step()



test_loss = evaluate(best_model, test_data)

print('=' * 80)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


