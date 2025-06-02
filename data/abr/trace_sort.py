import os
import shutil


def copy_files_in_batches(source_dir, destination_dir_base, batch_size=400):
    # 获取当前目录下所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 按文件名升序排序
    files.sort(key=lambda x:int(x.split('CUHK')[1]))

    # 创建批次并复制文件
    for i in range(5):
        # 定义目标文件夹名称
        destination_dir = f"{destination_dir_base}{i+1}/"

        # 如果目标文件夹不存在则创建
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # 复制当前批次的文件
        for file in files[i*batch_size : (i+1) * batch_size]:
            shutil.copy(os.path.join(source_dir, file), os.path.join(destination_dir, file))

        print(f"Copied batch {i // batch_size + 1} to {destination_dir}")


# 使用示例
source_directory = "/home/ubuntu/Whr/EAS/new_val/data/abr/NewFile-HighDensity-CUHK-train/"  # 当前目录
destination_directory_base = "/home/ubuntu/Whr/EAS/new_val/data/abr/classify_2/"  # 基础目标目录名称
copy_files_in_batches(source_directory, destination_directory_base)
