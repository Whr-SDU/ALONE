import matplotlib.pyplot as plt


# 左边子图数据
labels_left = ['Mean', 'Var', 'Time', 'Loc']
values_left = [9.6, 9.39, 8.62, 9.02]
errors_left = [0.74, 0.68, 0.76, 0.68]
colors_left = ['#4485c7', '#a7c0df', '#A0D0D0', '#D5aabe']
hatches_left = ['/', '\\', 'x', '-']

labels_right = ['1D','2D', '3D','4D']
values_right = [9.6, 9.73, 9.37, 9.16]
errors_right = [0.74, 0.69,0.77,0.71]
colors_right = ['#4485c7','#d0dd97', '#e6b745','#f09ba0']
hatches_right = ['/', '+', 'o','.']


# 创建左右两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# 左边子图
bars1 = ax1.bar(labels_left, values_left, yerr=errors_left, color=colors_left, edgecolor='black', capsize=20 ,error_kw=dict(elinewidth=2))
for i, bar in enumerate(bars1):
    bar.set_hatch(hatches_left[i])
    bar.set_label(labels_left[i])

# 添加标题和标签
# plt.title('示例柱状图')
ax1.set_xlabel('Individual network features',fontsize=22)
ax1.set_ylabel('QoE Value',fontsize=22)
ax1.set_ylim(7.7, 10.5)
ax1.tick_params(axis='y', labelsize=22)
ax1.set_xticks(range(len(labels_left)))
ax1.set_xticklabels(labels_left, fontsize=22,rotation=0)
# 左子图 legend


# 右边子图
bars2 = ax2.bar(labels_right, values_right, yerr=errors_right, color=colors_right, edgecolor='black', capsize=20 ,error_kw=dict(elinewidth=2))
for i, bar in enumerate(bars2):
    bar.set_hatch(hatches_right[i])
    bar.set_label(labels_right[i])
ax2.set_xlabel('Combined network features',fontsize=22)
ax2.set_ylabel('QoE Value',fontsize=22)
ax2.set_ylim(7.7, 10.5)
ax2.tick_params(axis='y', labelsize=22)
ax2.set_xticks(range(len(labels_right)))
ax2.set_xticklabels(labels_right, fontsize=22,rotation=0)

# 开启右边子图的 y 轴刻度标签
ax2.tick_params(labelleft=True)



plt.subplots_adjust(wspace=0.3)  # 把子图横向间距调大一些，默认大概是 0.2
# 保存到当前目录（支持 PNG、JPG、PDF、SVG 等）
plt.savefig("F_4.svg", dpi=600, bbox_inches='tight')  # dpi 可调清晰度
plt.close()