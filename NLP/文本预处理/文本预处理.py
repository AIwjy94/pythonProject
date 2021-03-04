# import jieba
# content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
# wen = jieba.cut(content, cut_all=False)  # cut_all默认为False
# print(wen)
#
# wen1 = jieba.lcut(content, cut_all=False)
# print(wen1)


# import hanlp
# 加载CTB_CONVSEG预训练模型进行分词任务
# tokenizer = hanlp.load('CTB6_CONVSEG')
# h1 = tokenizer("工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作")
# ['工信处', '女', '干事', '每', '月', '经过', '下', '属', '科室', '都', '要', '亲口', '交代', '24口', '交换机', '等', '技术性', '器件', '的', '安装', '工作']
# print(h1)

# import hanlp
# 加载中文命名实体识别的预训练模型MSRA_NER_BERT_BASE_ZH
# recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
# 这里注意它的输入是对句子进行字符分割的列表, 因此在句子前加入了list()
# >>> list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美 国纽约现代艺术博物馆参观。')
# ['上', '海', '华', '安', '工', '业', '（', '集', '团', '）', '公', '司', '董', '事', '长', '谭', '旭', '光', '和', '秘', '书', '张', '晚', '霞', '来', '到', '美', '国', '纽', '约', '现', '代', '艺', '术', '博', '物', '馆', '参', '观', '。']
# print(recognizer(list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。')))
# [('上海华安工业（集团）公司', 'NT', 0, 12), ('谭旭光', 'NR', 15, 18), ('张晚霞', 'NR', 21, 24), ('美国', 'NS', 26, 28), ('纽约现代艺术博物馆', 'NS', 28, 37)]

# 返回结果是一个装有n个元组的列表, 每个元组代表一个命名实体, 元组中的每一项分别代表具体的命名实体, 如: '上海华安工业（集团）公司'; 命名实体的类型, 如: 'NT'-机构名; 命名实体的开始索引和结束索引, 如: 0, 12.



# 导入用于对象保存与加载的joblib
from sklearn.externals import joblib
# 导入keras中的词汇映射器Tokenizer
from keras.preprocessing.text import Tokenizer
# 假定vocab为语料集所有不同词汇集合
vocab = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿 晗"}
# 实例化一个词汇映射器对象
t = Tokenizer(num_words=None, char_level=False)
# 使用映射器拟合现有文本数据
t.fit_on_texts(vocab)

for token in vocab:
    zero_list = [0]*len(vocab)
    # 使用映射器转化现有文本数据, 每个词汇对应从1开始的自然数
    # 返回样式如: [[2]], 取出其中的数字需要使用[0][0]
    token_index = t.texts_to_sequences([token])[0][0] - 1
    zero_list[token_index] = 1
    print(token, "的one-hot编码为:", zero_list)

# 使用joblib工具保存映射器, 以便之后使用
tokenizer_path = "./Tokenizer"
joblib.dump(t, tokenizer_path)