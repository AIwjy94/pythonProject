import numpy as np
import pandas as pd

arr1 = np.array([[1, 2, 3, 2, 1, 4], [5, 6, 1, 2, 3, 1]])
arr2 = np.array([[1], [3]])

# print(arr1 + arr2)

# print(pd.Series(np.arange(10)))
# print(pd.Series([6.7, 5.6, 3, 10, 2], index=[1, 2, 3, 4, 5]))


# color_count = pd.Series({'red':100, 'blue':200, 'green': 500, 'yellow':1000})
# print(color_count)
# print(color_count.index)


# print(pd.DataFrame(np.random.randn(2, 3)))


score = np.random.randint(40, 100, (10, 5))

# print(score)

score_df = pd.DataFrame(score)

# print(score_df)

subjects = ["语文", "数学", "英语", "政治", "体育"]

# 构造列索引序列
stu = ['同学' + str(i) for i in range(score_df.shape[0])]

# 添加行索引
data = pd.DataFrame(score, columns=subjects, index=stu)

data.T

print(data.T)


