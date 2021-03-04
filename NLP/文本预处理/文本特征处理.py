# 一般n-gram中的n取2或者3, 这里取2为例
ngram_range = 2

def create_ngram_set(input_list):
    """
    description: 从数值列表中提取所有的n-gram特征
    :param input_list: 输入的数值列表, 可以看作是词汇映射后的列表,
                       里面每个数字的取值范围为[1, 25000]
    :return: n-gram特征组成的集合

    eg:
    create_ngram_set([1, 4, 9, 4, 1, 4])
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    """
    # create_ngram_set([1, 4, 9, 4, 1, 4])

    return set(zip(*[input_list[i:] for i in range(ngram_range)]))


input_list = [1, 3, 2, 1, 5, 3]
res = create_ngram_set(input_list)
print(res)


from keras.preprocessing import sequence

# cutlen根据数据分析中句子长度分布，覆盖90%左右语料的最短长度.
# 这里假定cutlen为10
cutlen = 10

def padding(x_train):
    """
    description: 对输入文本张量进行长度规范
    :param x_train: 文本的张量表示, 形如: [[1, 32, 32, 61], [2, 54, 21, 7, 19]]
    :return: 进行截断补齐后的文本张量表示
    """
    # 使用sequence.pad_sequences即可完成
    return sequence.pad_sequences(x_train, cutlen)



# 假定x_train里面有两条文本, 一条长度大于10, 一天小于10
x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
           [2, 32, 1, 23, 1]]

res = padding(x_train)
print('')
print('')
print(res)