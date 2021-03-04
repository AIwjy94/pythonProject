# 第一步构建数据生成器


import transformer架构解析

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F



from pyitcast.transformer_utils import Batch

def data_generator(V, batch, num_batch):

    for i in range(num_batch):

        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))

        data[:, 0] = 1

        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        yield Batch(source, target)

V = 11
batch = 20
num_batch = 30

if __name__ == '__main__':
    res = data_generator(V, batch, num_batch)
    print(res)



# 第二步:获得transformer模型及其优化器和损失函数
from pyitcast.transformer_utils import get_std_opt

from pyitcast.transformer_utils import LabelSmoothing

from pyitcast.transformer_utils import SimpleLossCompute

from transformer架构解析 import make_model

model = make_model(V, V, N=2)

model_optimizer = get_std_opt(model)

criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

# 标签平滑示例
import matplotlib.pyplot as plt
crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)

predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0]]))

target = Variable(torch.LongTensor([2, 1, 0]))

crit(predict, target)

plt.imshow(crit.true_dist)
plt.show()

# 第三步：运行模型进行训练和评估
from pyitcast.transformer_utils import run_epoch

def run(model, loss, epochs=10):

    for epoch in range(epochs):

        model.train()

        run_epoch(data_generator(V, 8, 20), model, loss)

        model.eval()

        run_epoch(data_generator(V, 8, 5), model, loss)

# epochs = 10



# 第四步：使用模型进行贪婪解码
from pyitcast.transformer_utils import greedy_decode

def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()

        run_epoch(data_generator(V, 8, 20), model, loss)

        model.eval()

        run_epoch(data_generator(V, 8, 5), model, loss)

    model.eval()

    source = Variable(torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]]))

    source_mask = Variable(torch.ones(1, 1, 10))

    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == '__main__':
    run(model, loss)



