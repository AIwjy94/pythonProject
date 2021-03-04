# 从io工具包导入open方法
from io import open
# 用于字符规范化
import unicodedata
# 用于正则表达式
import re
# 用于随机生成数据
import random
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch中预定义的优化方法工具包
from torch import optim
# 设备选择，我们可以选择在cuda或者cpu上运行你的代码
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 第二步对持久化文件中数据进行处理，以满足模型训练要求
# 将指定语言中的词汇映射成数值
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1

class Lang:
    def __init__(self, name):
        '''初始化函数中参数name代表传入某种语言的名字'''
        # 将name转入类中
        self.name = name
        # 初始化词汇对应自然数值的字典
        self.word2index = {}
        # 初始化自然数值对应词汇的字典，其中0，1对应的SOS，和EOS已经在里面了
        self.index2word = {0: 'SOS', 1: 'EOS'}
        # 初始化词汇对应的自然数索引，这里从2开始，因为0，1已经被开始和结束标志占用了
        self.n_words = 2

    def addSentence(self, sentence):
        '''添加句子函数，即将句子转化为对应的数值序列，输入参数sentence是一条句子'''
        # 根据一般国家的语言特性（我们这里研究的语言都是以空格分割单词）
        # 对句子进行分割，得到对应的词汇列表
        for word in sentence.split(' '):
            # 然后调用addWord进行处理
            self.addWord(word)

    def addWord(self, word):
        '''添加词汇函数，即将词汇转化为对应的数值，输入参数word是一个单词'''
        # 首先判断word是否已经在self.word2index字典的key中
        if word not in self.word2index:
            # 如果不在，则将这个词加入其中，并为他对应一个数值，即self.n_words
            self.word2index[word] = self.n_words
            # 同时也将他的反转形式加入到self.index2word中
            self.index2word[self.n_words] = word
            # self.n_words一旦被占用之后，逐次加1，变成新的self.n_words
            self.n_words += 1

# name = 'eng'

# sentence = 'hello I am Jay'
#
# eng1 = Lang(name)
# eng1.addSentence(sentence)
# print('word2index:', eng1.word2index)
# print('index2word:', eng1.index2word)
# print('n_words:', eng1.n_words)



# 字符规划化
# 将unicode转化为Ascii，我们可以认为是去掉一些语言中的重音标记：Ślusàrski
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) !='Mn')

def normalizeString(s):
    '''字符串规范化函数，参数s代表传入的字符串'''
    # 使字符变为小写并去除两侧空白符，再使用unicodeToAscii去掉重音标记
    s = unicodeToAscii(s.lower().strip())
    # 在.!?前加一个空格
    s = re.sub(r"([.!?])", r"\1", s)
    # 使用正则表达式将字符串中不是大小字母和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ",s)
    return s

# 输入参数
# s = "Are you kidding me?"
# 调用
# nsr = normalizeString(s)
# print(nsr)


# 将持久化文件中的数据加载到内存，并实列化类lang
data_path = './data/eng-fra.txt'

def readLangs(lang1, lang2):
    '''读取语言函数，参数lang1是源语言的名字，参数lang2是目标语言的名字'''
    '''返回对应的class lang对象，以及语言对列表'''
    # 从文件中读取语言对并以/n划分存到列表lines中
    lines = open(data_path, encoding='utf-8').\
        read().strip().split('\n')
    # 对lines列表中的句子进行标准化处理，并以\t进行再次划分，形成子列表，也就是语言对
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # 然后分别将语言名字传入Lang类中，获得对应的语言对象，返回结果
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

# 输入参数
lang1 = 'eng'
lang2 = 'fra'

# 调用
input_lang, output_lang, pairs = readLangs(lang1, lang2)
print('input_lang:', input_lang)
print('output_lang:', output_lang)
print('pairs中的前五个:', pairs[:5])


# 过滤出符合我们要求的语言对
MAX_LENGTH = 10

# 设置带有指定前缀的语言特征数据作为训练数据
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    '''语言对过滤函数，参数p代表输入的语言对，如['she is afraid.', 'elle malade.']'''
    # p[0]代表英语句子，对他进行划分，他的长度应小于最大长度MAX_LENGTH并且要以指定的前缀开头
    # p[1]代表法文句子，对他进行划分，他的长度应小于最大长度MAX_LENGTH
    return len(p[0].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes) and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    '''对多个语言对列表进行过滤，参数pairs代表语言对组成的列表，简称语言列表'''
    # 函数中直接遍历列表中的每个语言对并调用filterPair即可
    return [pair for pair in pairs if filterPair(pair)]


# 调用
fpairs = filterPairs(pairs)
print('过滤后的pairs前五个：', fpairs[:5])

# 对以上数据准备函数进行整合，并使用类Lang对语言对进行数值映射
def prepareData(lang1, lang2):
    '''数据准备函数，完成将所有字符串数据向数值型数据对的映射以及过滤语言对'''
    # 参数lang1, lang2分别代表源语言和目标语言的名字
    # 首先通过realLang函数获得input_lang, output_lang对象，以及字符串类型的语言对列表
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    # 对字符串类型的语言对列表进行过滤操作
    pairs = filterPairs(pairs)
    # 对过滤后的语言对列表进行遍历
    for pair in pairs:
        # 并使用input_lang和output_lang的addSentence方法对其进行数值映射
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    # 返回数值映射后的对象，和过滤后语言对
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra')
print('input_n_words:', input_lang.n_words)
print('output_n_words:', output_lang.n_words)
print(random.choice(pairs))


# 将语言对转化为模型输入需要的张量
def tensorFromSentence(lang, sentence):
    '''将文本句子转换为张量，参数lang代表传入的Lang的实列化对象，sentence是预转换的句子'''
    # 对句子进行分割并遍历每一个词汇，然后使用lang的word2index方法找到他对应的索引
    # 这样就得到了该句子对应的数值列表
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    # 然后加入句子结束标志
    indexes.append(EOS_token)
    # 将其使用torch.tensor封装成张量，并改变他的形状为nx1,以方便后续计算
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromPair(pair):
    '''将语言对转换为张量对，参数pair为一个语言对'''
    # 调用tensorFromSentence分别将源语言和目标语言分别处理，获得对应的张量表示
    inpput_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    # 最后返回他们组成的元组
    return (inpput_tensor, target_tensor)


# 输入参数
# 取pairs的第一条
pair = pairs[0]

# 调用
pair_tensor = tensorFromPair(pair)
print(pair_tensor)

# 构建基于GRU的编码器和解码器
# 构建基于GRU的编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        '''它的初始化参数有两个，input_size代表解码器的输入尺寸即源语言的词表大小，
        hidden_size代表GRU的隐层节点数，也代表词嵌入纬度，同时又是GRU的输入尺寸'''
        super(EncoderRNN, self).__init__()
        # 将参数hidden_size传入类中
        self.hidden_size = hidden_size
        # 实列化nn中预定义的Embedding层，它的参数分别是input_size, hidden_size
        # 这里的词嵌入纬度即hidden_size
        # nn.Embedding的演示在该代码下方
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 然后实列化nn中预定义的GRU层，它的参数是hidden_size
        # nn.GRU的演示在该代码下方
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        '''编码器前向逻辑函数中参数有两个，input代表源语言的Embedding层输入张量，hidden代表编码器层gru的初始隐层张量'''
        # 将输入张量进行embedding操作，并使其形状变为（1，1，-1），-1代表自动计算纬度
        # 理论上，我们的编码器每次只以一个词作为输入，因此词汇映射后的尺寸应该是[1, embedding]
        # 而这里转换成三维的原因是因为torch中预定义gru必须使用三维张量作为输入，因此我们扩展了一个纬度
        output = self.embedding(input).view(1, 1, -1)
        # 然后将embedding层的输出和传入的初始hidden作为gru的输入传入其中
        # 获得最终gru的输出output和对应的隐层张量hidden，并返回结果
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        '''初始化隐层张量函数'''
        # 将隐层张量初始化成为1x1xself.hidden_size大小的0张量
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 实列化参数
# hidden_size = 25
# input_size = 20

# 输入参数
# pair_tensor[0]代表源语言即英文的句子，pair_tensor[0][0]代表句子中的第一个词
# input = pair_tensor[0][0]
# 初始化第一个隐层张量，1x1xhidden_size的0张量
# hidden = torch.zeros(1, 1, hidden_size)

# 调用
# encoder = EncoderRNN(input_size, hidden_size)
# encoder_output, hidden = encoder(input, hidden)
# print(encoder_output)


# 构建GRU解码器
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        '''初始化函数有两个参数，hidden_size代表解码器中GRU的输入尺寸，也是它的隐层节点数
        output_size代表整个解码器的输出尺寸，也是我们希望得到的指定尺寸即目标语言的词表大小'''
        super(DecoderRNN, self).__init__()
        # 将hidden_size传入到类中
        self.hidden_size = hidden_size
        # 实列化一个nn中的Embeddding层对象，它的参数output这里表示目标语言的词表大小
        # hidden_size表示目标语言的词嵌入纬度
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 实列化GRU对象，输入参数都是hidden_size，代表它的输入尺寸和隐层节点数相同
        self.gru = nn.GRU(hidden_size, hidden_size)
        # 实列化线性层，对GRU的输出做线性变化，获我们希望的输出尺寸output_size
        # 因此它的两个参数分别是hidden_size， output_size
        self.out = nn.Linear(hidden_size, output_size)
        # 最后使用softmax进行处理，以便于分类
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        '''解码器的前向逻辑函数中，参数有两个，input代表目标语言的Embedding层输入张量
        hidden代表解码器GRU的初始隐层张量'''
        # 将输入张量进行embedding操作，并使其形状变为（1，1，-1），-1代表自动计算纬度
        # 原因和解码器相同，因为torch预定义的GRU层只接受三维张量作为输入
        output = self.embedding(input).view(1, 1, -1)
        # 然后使用relu函数对输出进行处理，根据relu函数的特性，将使embedding矩阵更稀疏，以防止过拟合
        output = F.relu(output)
        # 接下来，将把embedding的输出以及初始化的hidden张量传入到解码器gru中
        output, hidden = self.gru(output, hidden)
        # 因为GRU输出的output也是三维张量，第一维没有意义，因此可以通过output[0]来降维
        # 再传给线性层做变换，最后用softmax处理以便于分类
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        '''初始化隐层张量函数'''
        # 将隐层函数初始化为1x2xself.hidden_size大小的0张量
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 实列化参数
# hidden_size = 25
# output_size = 10

# 输入参数
# pair_tensor[1]代表目标语言即法文的句子，pair_tensor[1][0]代表句子中的第一个词
# input = pair_tensor[1][0]
# 初始化第一个隐层张量，1x1xhidden_size的0张量
# hidden = torch.zeros(1, 1, hidden_size)

# 调用
# decoder = DecoderRNN(hidden_size, output_size)
# output, hidden = decoder(input, hidden)
# print(output)

# 构建基于GRU和Attention的解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        '''初始化函数中的参数有4个，hidden_size代表解码器中GRU的输入尺寸，也是它的隐层节点数
        output_size代表整个解码器的输出尺寸，也是我们希望得到的指定尺寸即目标语言的词表大小
        dropout_p代表我们使用dropout层时的置零比率，默认0.1,max_length代表句子的最大长度'''

        super(AttnDecoderRNN, self).__init__()
        # 将一下参数传入类中
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 实列化一个Embedding层，输入参数是self.output_size和self.hidden_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)


        
        


