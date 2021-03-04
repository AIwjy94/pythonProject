# 假设取两条已经存在的正样本和两条负样本
# 将基于这四条样本产生新的同标签的四条样本
p_sample1 = "酒店设施非常不错"
p_sample2 = "这家价格很便宜"
n_sample1 = "拖鞋都发霉了, 太差了"
n_sample2 = "电视不好用, 没有看到足球"

# 导入google翻译接口工具
from googletrans import Translator
# 实例化翻译对象
translator = Translator()
# 进行第一次批量翻译, 翻译目标是韩语
translations = translator.translate([p_sample1, p_sample2, n_sample1, n_sample2], dest='ko')
# 获得翻译后的结果
ko_res = list(map(lambda x: x.text, translations))
# 打印结果
print("中间翻译结果:")
print(ko_res)


# 最后在翻译回中文, 完成回译全部流程
translations = translator.translate(ko_res, dest='zh-cn')
cn_res = list(map(lambda x: x.text, translations))
print("回译得到的增强数据:")
print(cn_res)