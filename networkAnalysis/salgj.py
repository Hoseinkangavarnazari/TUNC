import matplotlib.pyplot as plt
import numpy as np

# 设置数据
groups = 9
bars_per_group = 8

# 生成随机数据
np.random.seed(0)
data = np.random.rand(groups, bars_per_group)

# 生成一个足够大的图形
plt.figure(figsize=(15, 8))

# 创建柱形图
bar_width = 0.1
for i in range(groups):
    plt.bar(np.arange(bars_per_group) + i*bar_width, data[i], width=bar_width, label=f'Group {i+1}')

# 添加图例
plt.legend()

# 设置坐标轴标签
plt.xlabel('Bar Number within Group')
plt.ylabel('Value')
plt.title('Bar Chart with 9 Groups, Each Having 8 Bars')

# 调整 x 轴刻度，以清晰显示每组的标签
plt.xticks(np.arange(bars_per_group) + bar_width*(groups-1)/2, [f'Bar {i+1}' for i in range(bars_per_group)])

# 显示图形
plt.show()
