# 从io到入文件打开方法
from io import open
# 帮助使用正则表达式进行子目录的查询
import glob
import os
# 用于获得常见字母及字符规范化
import string
import unicodedata
# 导入随机工具random
import random
# 导入时间和数学工具包
import time
import math
# 导入torch
import torch
# 导入nn准备构建模型
import torch.nn as nn
# 引入制图工具包
import matplotlib.pyplot as plt


# 第二步：对data文件中数据进行处理，满足训练要求
# 获取常用的字符数量
# 获取所有字符包括字母和常用标点
all_letters = string.ascii_letters + " .,;'"

# 获取常用字符数量
n_letters = len(all_letters)

# print("n_letter:", n_letters)

# 字符规范化之unicode转Ascii函数
# 关于编码问题我们暂且不去考虑
# 我们认为这个函数的作用就是去掉一些语言中的重音标记
# 如: Ślusàrski ---> Slusarski
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn'
                    and c in all_letters)
# s = "Ślusàrski"
# a = unicodeToAscii(s)
# print(a)

# 构建一个持久化文件中读取内容到内存的函数

# data_path = "/Users/wjy/Pycharm/pythonProject/NLP/RNN/data/names/"  绝对路径有问题
data_path = "./data/names/"

def readLines(filename):
    # 打开指定文件并读取所有内容，使用strip()去除两侧空白符，然后以'\n'进行切分
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 对应每一个lines列表中的名字进行Ascii转换，使其规范化，最后返回一个名字列表
    return [unicodeToAscii(line) for line in lines]

# filename是数据集中某个具'体的文件，我们这里选择Chinese.txt
filename = data_path + "Chinese.txt"
lines = readLines(filename)
# print(lines)



# 构建人名类别（所属的语音）列表与人名对应关系字典
# 构建的category_lines形如：{"English":["Lily", "Susan", "Kobe"], "Chinese":["Zhang San", "Xiao Ming"]}
category_lines = {}

# all_categories形如： ["English",...,"Chinese"]
all_categories = []

# 读取指定路径下的txt文件，使用glob， path中的可以使用正则表达式
for filename in glob.glob(data_path + '*.txt'):
    # 获取每个文件的文件名，就是对应的名字类别
    category = os.path.splitext(os.path.basename(filename))[0]
    # 将其琢一装到all_categories列表中
    all_categories.append(category)
    # 然后读取每个文件的内容，形成名字列表
    lines = readLines(filename)
    # 按照对应的类别，将名字列表写入到category_lines字典中
    category_lines[category] = lines

# 查看类别总数
n_categories = len(all_categories)
print('n_categories:', n_categories)

# 随便查看其中的一些内容
# print(category_lines['Italian'][:5])



# 将人名转化为对应onehot张量表示
def lineToTensor(line):
    '''将人名转化为对应onehot张量表示，参数line是输入的人名'''
    # 首先初始化一个0张量，他的形状(len(line), 1, n_letters)
    # 代表人名中的每个字母用一个1xn_letters的张量表示
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历这个人名中的每个字符索引和字符
    for li, letter in enumerate(line):
        # 使用字符串方法find找到每个字符在all_letters中的索引
        # 他是我们生成onehot张量中1的索引位置
        tensor[li][0][all_letters.find(letter)] = 1

    # 返回结果
    return tensor

line = "Bai"
line_tensor = lineToTensor(line)
print('line_tensot:', line_tensor)

# 第三步 构建RNN模型

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers)

        self.linear = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        rr, hn = self.rnn(input, hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 构建LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化函数的参数与传统RNN相同"""
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实列化预定义的nn.LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 实列化nn.Linear，这个线性层用于将nn.RNN的输出纬度转化为指定的输出结果
        self.linear = nn.Linear(hidden_size, output_size)
        # 实列化nn中预定义的Softmax层，用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c):
        input = input.unsqueeze(0)
        rr, (hn, c) = self.lstm(input, (hidden, c))
        return self.softmax(self.linear(rr)), hn, c

    def initHiddenAndC(self):
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c


# 构建GRU模型
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        rr, hn = self.gru(input, hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)



input_size = n_letters

n_hidden = 128

output_size = n_categories


input = lineToTensor('B').squeeze(0)

hidden = c = torch.zeros(1, 1, n_hidden)

rnn = RNN(n_letters, n_hidden, n_categories)
lstm = LSTM(n_letters, n_hidden, n_categories)
gru = GRU(n_letters, n_hidden, n_categories)

rnn_output, next_hidden = rnn(input, hidden)
print('rnn:', rnn_output)
lstm_output, next_hidden, c = lstm(input, hidden, c)
print('lstm:', lstm_output)
gru_output, next_hidden = gru(input, hidden)
print('gru:', gru_output)


# 第四步构建训练函数并进行训练
def categoryFromOutput(output):
    '''从输出结果中获得指定类别，参数为输出张量output'''
    # 从输出张量中返回最大的值和索引对象，我们这里主要需要这个索引
    top_n, top_i = output.topk(1)
    # top_i对象中取出索引的值
    category_i = top_i[0].item()
    # 根据索引值获得对应语言类别，返回语言类别和索引值
    return all_categories[category_i], category_i


output = gru_output

category, category_i = categoryFromOutput(output)
print('category:', category)
print('category_i:', category_i)


# 随机生成训练数据
def randomTrainingExample():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)

    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


# 调用
for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line, '/ category_tensor =', category_tensor)



# 构建传统RNN训练函数
criterion = nn.NLLLoss()

learning_rate = 0.005

def trainRNN(category_tensor, line_tensor):

    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)


    loss = criterion(output.squeeze(0), category_tensor)

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


# 构建LSTM训练函数
def trainLSTM(category_tensor, line_tensor):
    hidden, c = lstm.initHiddenAndC()
    lstm.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()

    for p in lstm.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


# 构建RGU训练函数
def trainGRU(category_tensor, line_tensor):
    hidden = gru.initHidden()
    gru.zero_grad()
    for i in range(line_tensor()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()

    for p in gru.parameters():
        p.data.add_(learning_rate, p.grad.data)
    return output, loss.item()

# 构建时间计算函数
def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)

since = time.time() - 10 * 60

# 调用
period = timeSince(since)
print(period)


# 构建训练过程的日志打印函数
# 设置训练迭代次数
n_iters = 1000
# 设置结果的打印间隔
print_every = 50
# 设置绘制损失曲线上的制图间隔
plot_every = 10

def train(train_type_fn):
    all_losses = []

    start = time.time()

    current_loss = 0

    for iter in range(1, n_iters + 1):

        category, line, category_tensor, line_tensor = randomTrainingExample()

        output, loss = train_type_fn(category_tensor, line_tensor)

        current_loss += loss

        if iter % print_every == 0:

            guess, guess_i = categoryFromOutput(output)

            correct = '✓' if guess == category else '✗ (%s)' % category

            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        if iter % plot_every == 0:

            all_losses.append(current_loss / plot_every)

            current_loss = 0

    return all_losses, int(time.time() - start)

# 调用train函数, 分别进行RNN, LSTM, GRU模型的训练
# 并返回各自的全部损失, 以及训练耗时用于制图
all_losses1, period1 = train(trainRNN)
all_losses2, period2 = train(trainLSTM)
all_losses3, period3 = train(trainGRU)

# 绘制损失对比曲线, 训练耗时对比柱张图
# 创建画布0
plt.figure(0)
# 绘制损失对比曲线
plt.plot(all_losses1, label="RNN")
plt.plot(all_losses2, color="red", label="LSTM")
plt.plot(all_losses3, color="orange", label="GRU")
plt.legend(loc='upper left')


# 创建画布1
plt.figure(1)
x_data=["RNN", "LSTM", "GRU"]
y_data = [period1, period2, period3]
# 绘制训练耗时对比柱状图
plt.bar(range(len(x_data)), y_data, tick_label=x_data)


# 第五步 构建评估函数并进行预测

# 构建传统RNN评估函数
def evaluateRNN(line_tensor):
    '''评估函数，和训练函数逻辑相同，参数line_tensor代表名字的张量表示'''
    # 初始化隐层张量
    hidden = rnn.initHidden()
    # 将评估函数line_tenor的每个字符逐个传入rnn之中
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output.squeeze(0)

# 构建LSTM评估函数
def evaluateLSTM(line_tensor):
    # 初始化隐层张量和细胞状态张量
    hidden, c = lstm.initHiddenAndC()
    # 将评估数据line_tensor的每个字符逐个传入lstm之中
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    return output.squeeze(0)

# 构建GRU评估函数
def evaluateGRU(line_tensor):
    hidden = gru.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    return output.squeeze(0)

line = "Bai"
line_tensor = lineToTensor(line)

rnn_output = evaluateRNN(line_tensor)
lstm_output = evaluateLSTM(line_tensor)
gru_output = evaluateGRU(line_tensor)
print("rnn_output:", rnn_output)
print("gru_output:", lstm_output)
print("gru_output:", gru_output)

def predict(input_line, evaluate, n_predictions=3):

    print('\n> %s' % input_line)

    with torch.no_grad():
        # 使输入的名字转换为张量表示，并使用evaluation函数获得预测输出
        output = evaluate(lineToTensor(input_line))

        # 从预测的输出中取前3个最大的值及其索引
        topv, topi = output.topk(n_predictions, 1, True)
        # 创建盛装结果列表
        predictions = []
        # 遍历n_predictions
        for i in range(n_predictions):

            # 从topv中取出的output值
            value = topv[0][i].item()
            # 取出索引并找到对应的类别
            category_index = topi[0][i].item()
            # 打印output的值，和对应的类别
            print('(%.2f) %s' % (value, all_categories[category_index]))
            # 将结果装进predictions
            predictions.append([value, all_categories[category_index]])

for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGRU]:
    print('-'*18)
    predict('Dovesky', evaluate_fn)
    predict('Jackson', evaluate_fn)
    predict('Satoshi', evaluate_fn)
