import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


BITS_IN_BYTE = 8.0
INIT_CHUNK = 0
M_IN_K = 1000.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
VIDEO_BIT_RATE = [200., 800., 2200., 5000., 10000.,18000.,32000.,50000.]
K_IN_M = 1000.0
SMOOTH_P = 1
COLOR_MAP = plt.cm.rainbow#plt.cm.jet #nipy_spectral, Set1,Paired plt.cm.rainbow#


rewards = [[-0.3,1.81,5.24,14.15,19.1],
		   [-8.16,-1.08,9.88,18.49,21.2],[-9.5,-3.81,2.63,20.25,32.6],
		   [-5.59,2.9,8.07,21.5,28],[-1.41,5.08,9.51,23.55,31.59]]
volatile = [[4.7, 5.75, 5.22,   5.6, 5.49],
			[ 3.83, 6.02, 4.47,  4.49, 5.67],[4.26, 5.73, 5.3,  4.67, 4.69],
			[4.36, 5.97, 4.28,  5.21, 6.14],[ 4.84, 5.92,4.98, 4.95, 6.05]]



NORM = False #True 


def save_csv(data, file_name):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(file_name,index=False,sep=',')

def main():




	labels = ['Class 1','Class 2','Class 3','Class 4','Class 5']


	QoE_num = ['Specialist 1','Specialist 3','Specialist 5','Initial model','Generalist']
	# 绘图配置
	x = np.arange(len(labels))  # x轴
	# x = np.array([0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5])  # 手动指定每个 Class 的 x 位置

	width = 0.14  # 每组的宽度
	# 颜色
	colors = ['#547BB4','#DD7C4F','#629C35','#b55489','tomato']
	fig, ax = plt.subplots(figsize=(17, 6))
	# 不同方法对应的标记样式
	markers = [ 'v','^','<','o', '*']  # 圆形、下三角、上三角、正方形、五边形、菱形
	# 绘制每种方法的误差条图
	for i, (qoe_num, marker) in enumerate(zip(QoE_num,markers)):
		ax.errorbar(
			x + i * width - (len(QoE_num) - 1) / 2 * width  ,  # 调整每组的x位置
			[rewards[i][method] for method in range(len(labels))],  # 均值
			yerr=[volatile[i][method] for method in range(len(labels))],  # 误差
			fmt=marker, markersize=25, label=qoe_num, capsize=12,  # 设置误差棒帽子边框宽度, # 样式：圆点+误差条
			linestyle='None',  color=colors[i], ecolor='black', # 禁用线条，给每种方法不同颜色
			markeredgewidth=1, markeredgecolor='black'  # # 边框宽度和颜色
		)
	
	for i in range(len(x)-1):
		plt.plot([x[i] + 5 * 0.1,x[i] + 5 * 0.1],[-15,40], linewidth=2, linestyle="--",color = 'grey')

	# ax.axvline((x[0] + x[1]) / 2, color='grey', linestyle='--', linewidth=1.5)
	# 添加图例、标签和刻度
	ax.grid(True)

	ax.set_xticks(x)
	ax.set_xticklabels(labels,fontsize=28)
	ax.set_ylabel('QoE Value',fontsize=28)
	ax.tick_params(axis='y', labelsize=28)
	ax.set_yticks(np.arange(-15, 40, 10))
	ax.set_ylim([-15, 40])
	ax.set_xlim([-0.5, 4.5])
	ax.legend(
		loc='upper right',  # 设置图例位置（顶部中间）
		ncol=2,  # 图例分为 2 列
		#markerscale = 0.6,
		fontsize=25,  # 图例字体大小
		bbox_to_anchor=(0.55, 1.0),
		markerscale = 0.8,  # 缩小标记比例
	)

	plt.savefig('F_3.svg', dpi=600, orientation='portrait', format='svg',transparent=False, bbox_inches='tight', pad_inches=0.1)

if __name__ == '__main__':
	main()
