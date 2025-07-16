import matplotlib.pyplot as plt
import numpy as np

# 定义颜色、花纹等
colors = ['#bbdbb3', '#91CCc0', '#f7ac53', '#9998ff', '#7fabd1', '#db7272']
hatch_patterns = ['/', '\\', '-', '.', 'x', '+']
x_labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '>0.5']


# schemes_show 和 schemes_label 需要你自己定义好
# 例子：

schemes_label = ['RobustMPC', 'Pensieve', 'Merina', 'Netllm', 'Genet', 'ANOLE']

# 四组数据
persent = [[78.3, 75.2, 72.77, 60.54, 47.91, 8.34], [79.34, 74.56, 74.88, 69.82, 55.11, 4.25],
           [76.22, 74.84, 74.33, 57.77, 47.3, -13.45], [80.0, 78.52, 76.43, 70.7, 57.44, 12.47],
           [74.99, 73.79, 69.09, 66.55, 54.07, 23.1], [84.16, 84.32, 83.21, 78.11, 61.85, 42.53]]
persent_err = [[6.53, 4.44, 7.11, 5.61, 7.85, 8.44], [6.75, 5.07, 8.1, 7.53, 7.34, 8.72],
               [7.11, 4.58, 7.9, 7.99, 8.48, 8.03], [6.76, 4.9, 7.87, 6.42, 8.71, 7.87],
               [5.6, 5.06, 7.91, 8.94, 7.15, 7.62], [6.29, 4.82, 7.81, 6.25, 7.81, 8.28]]

qua = [[23.68, 18.59, 15.02, 14.87, 13.52, 10.92], [21.98, 18.05, 14.58, 14.35, 12.72, 10.57],
       [23.67, 18.66, 15.17, 15.21, 13.59, 10.94], [22.23, 17.76, 14.43, 14.38, 12.55, 11.1],
       [20.85, 17.44, 14.08, 13.67, 12.33, 10.09], [23.65, 18.60, 14.70, 14.43, 13.02, 10.40]]
qua_err = [[3.75, 0.9, 1.28, 2.23, 1.53, 3.78], [3.33, 0.91, 1.28, 2.27, 1.56, 4.03],
           [3.66, 0.9, 1.28, 2.26, 1.53, 3.8], [3.54, 0.91, 1.31, 2.29, 1.49, 4.19],
           [3.29, 0.91, 1.23, 2.26, 1.55, 3.87], [3.32, 0.91, 1.3, 2.18, 1.48, 3.86]]

rebuf = [[0.99, 0.91, 1.58, 4.78, 6.71, 9.18], [0.66, 0.84, 1.13, 2.33, 3.2, 8.62],
         [0.92, 0.93, 1.46, 6.61, 7.34, 10.01], [1.02, 0.74, 0.92, 2.28, 3.45, 8.91],
         [0.66, 0.7, 1.13, 1.49, 3.18, 4.14], [0.35, 0.38, 0.55, 1.19, 2.08, 3.1]]
rebuf_err = [[0.54, 0.08, 0.13, 1.33, 1.12, 1.75], [0.04, 0.69, 0.42, 0.84, 1.67, 2.42],
             [0.45, 0.16, 0.26, 1.58, 1.51, 1.37], [0.52, 0.35, 0.35, 0.95, 1.33, 1.08],
             [0.04, 0.3, 0.35, 0.76, 0.97, 0.73], [0.08, 0.1, 0.05, 0.82, 0.94, 0.95]]


# 创建1行3列子图
fig, axs = plt.subplots(1, 3, figsize=(29, 7))
width = 0.14
x = np.arange(1, 7)
# 绘制第一个子图：Normalize QoE
for scheme in range(len(schemes_label)):
    axs[0].bar(x + (scheme - len(schemes_label) / 2) * width, persent[scheme], width,
               label=schemes_label[scheme], color=colors[scheme], hatch=hatch_patterns[scheme],
               edgecolor='black', yerr=persent_err[scheme], capsize=10)
axs[0].set_ylim(-29, 100)
axs[0].set_xticks(x- 0.1)
axs[0].set_xticklabels(x_labels, fontsize=22)
axs[0].set_ylabel('Normalize QoE', fontsize=22)
axs[0].set_xlabel('Cov level', fontsize=22)
axs[0].tick_params(axis='both', labelsize=22)
for i in range(len(x) - 1):
    axs[0].plot([x[i] + 4.2 * 0.1, x[i] + 4.2 * 0.1], [-25, 120], linestyle="--", color='grey', linewidth=2)
axs[0].legend(fontsize=20, loc='lower left',ncol=3, handletextpad=0.3, columnspacing=0.6)

# 绘制第二个子图：Bitrate
for scheme in range(len(schemes_label)):
    axs[1].bar(x + (scheme - len(schemes_label) / 2) * width, qua[scheme], width,
               label=schemes_label[scheme], color=colors[scheme], hatch=hatch_patterns[scheme],
               edgecolor='black', yerr=qua_err[scheme], capsize=10)
axs[1].set_ylim(0, 29)
axs[1].set_xticks(x-0.1)
axs[1].set_xticklabels(x_labels, fontsize=22)
axs[1].set_ylabel('Bitrate(Mbps)', fontsize=22)
axs[1].set_xlabel('Cov level', fontsize=22)
axs[1].tick_params(axis='both', labelsize=22)
for i in range(len(x) - 1):
    axs[1].plot([x[i] + 4.2 * 0.1, x[i] + 4.2 * 0.1], [0, 29], linestyle="--", color='grey', linewidth=2)
axs[1].legend(fontsize=20, loc='upper right',ncol=3, handletextpad=0.3, columnspacing=0.6)

# 绘制第三个子图：Rebuffer
for scheme in range(len(schemes_label)):
    axs[2].bar(x + (scheme - len(schemes_label) / 2) * width, rebuf[scheme], width,
               label=schemes_label[scheme], color=colors[scheme], hatch=hatch_patterns[scheme],
               edgecolor='black', yerr=rebuf_err[scheme], capsize=10)
axs[2].set_ylim(0, 12)
axs[2].set_xticks(x-0.1)
axs[2].set_xticklabels(x_labels, fontsize=22)
axs[2].set_ylabel('Rebuffer(sec)', fontsize=22)
axs[2].set_xlabel('Cov level', fontsize=22)
axs[2].tick_params(axis='both', labelsize=22)
for i in range(len(x) - 1):
    axs[2].plot([x[i] + 4.2 * 0.1, x[i] + 4.2 * 0.1], [0, 12], linestyle="--", color='grey', linewidth=2)
axs[2].legend(fontsize=20, loc='upper left',ncol=1, handletextpad=0.3, columnspacing=0.6)


plt.subplots_adjust(wspace=0.2, hspace=0.25)
fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.13)

plt.savefig( "F_2.svg", dpi=600)

