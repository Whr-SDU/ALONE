import matplotlib.pyplot as plt


# 左边子图数据
label_left = ['Normalize QoE','Training Time']
x_labels_left = ['2', '4', '6','8','10' ]
values_left = [0.906, 0.953, 0.982,0.998,1]
values_left_2 = [0.40, 0.54, 0.69,0.85,1]
errors_left = [0.032, 0.034, 0.030,0.033,0.029]
colors_left = ['#9dc3e7', '#9dc3e7', '#9dc3e7','#9dc3e7', '#9dc3e7']
hatches_left = ['','','','/',  '']

labels_right = ['With','Without']
values_right = [1, 0.979]
values_right_2 = [0.36, 1]
errors_right = [0.033, 0.036]
colors_right = ['#9dc3e7','#9dc3e7']
hatches_right = ['/', '']


labels_3 = ['0.05','0.1','0.5','1']
values_3 = [0.94, 0.98,1,0.992]
values_3_2 = [1,0.81,0.67,0.62]
errors_3 = [0.029, 0.031,0.032,0.030]
colors_3 = ['#9dc3e7', '#9dc3e7', '#9dc3e7','#9dc3e7', '#9dc3e7']
hatches_3 = ['','','/','']



fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(15, 6),gridspec_kw={'width_ratios': [1.3, 2, 2]})

bars2 = ax1.bar(labels_right, values_right, yerr=errors_right, color=colors_right, edgecolor='black', capsize=20 ,error_kw=dict(elinewidth=2))
for i, bar in enumerate(bars2):
    bar.set_hatch(hatches_right[i])
    # bar.set_label(labels_right[i])

ax1.set_xlabel('With or without Stage I',fontsize=22)
ax1.set_ylabel('Value',fontsize=22)
ax1.set_ylim(0.25, 1.05)
ax1.tick_params(axis='y', labelsize=22)
ax1.set_xticks(range(len(labels_right)))
ax1.set_xticklabels(labels_right, fontsize=20,rotation=0)
ax1.tick_params(labelleft=True)


# 折线图
x = list(range(len(labels_right)))  # x 坐标是 0, 1, 2, ...
ax1.plot(x, values_right_2, color='black', marker='o', linewidth=2, markersize=8)





bars1 = ax2.bar(x_labels_left, values_left, yerr=errors_left, color=colors_left, edgecolor='black', capsize=15 ,error_kw=dict(elinewidth=2))
for i, bar in enumerate(bars1):
    bar.set_hatch(hatches_left[i])
    # bar.set_label(x_labels_left[i])

# 折线图
x = list(range(len(x_labels_left)))  # x 坐标是 0, 1, 2, ...
ax2.plot(x, values_left_2, color='black', marker='o', linewidth=2, markersize=8)

# 添加标题和标签
# plt.title('示例柱状图')
ax2.set_xlabel('Number of experts',fontsize=22)
ax2.set_ylabel('Value',fontsize=22)
ax2.set_ylim(0.39, 1.05)
ax2.tick_params(axis='y', labelsize=22)
ax2.set_xticks(range(len(x_labels_left)))
ax2.set_xticklabels(x_labels_left, fontsize=22,rotation=0)





bars3 = ax3.bar(labels_3, values_3, yerr=errors_3, color=colors_3, edgecolor='black', capsize=15 ,error_kw=dict(elinewidth=2))
for i, bar in enumerate(bars3):
    bar.set_hatch(hatches_3[i])
    # bar.set_label(labels_3[i])
    if i ==0:
        bar.set_label(label_left[0])  # 只给第一个柱子设置图例标签


# 折线图
x = list(range(len(labels_3)))  # x 坐标是 0, 1, 2, ...
ax3.plot(x, values_3_2, color='black', marker='o', linewidth=2, markersize=8, label=label_left[1])

ax3.set_xlabel('Value of weight α',fontsize=22)
ax3.set_ylabel('Value',fontsize=22)
ax3.set_ylim(0.5, 1.05)
ax3.tick_params(axis='y', labelsize=22)
ax3.set_xticks(range(len(labels_3)))
ax3.set_xticklabels(labels_3, fontsize=22,rotation=0)

# 开启右边子图的 y 轴刻度标签
ax3.tick_params(labelleft=True)




fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=20, ncol=4)




plt.subplots_adjust(wspace=0.5)  # 把子图横向间距调大一些，默认大概是 0.2
# 保存到当前目录（支持 PNG、JPG、PDF、SVG 等）
plt.savefig("F_7.svg", dpi=600, bbox_inches='tight')  # dpi 可调清晰度
plt.close()