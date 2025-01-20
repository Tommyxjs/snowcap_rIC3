# -*- coding: utf-8 -*-
import subprocess
import re
import os

directory = './eval_sigcomm2021/topology_zoo/'
output_file = 'experiment_results_cdcl.txt'

# 确保输出文件是空的，或者创建新的
with open(output_file, 'w') as f:
    f.write('Filename\tRuntime\n')

# 遍历目录中的所有 .gml 文件
for filename in os.listdir(directory):
    if filename.endswith('.gml'):
        # 构造完整的文件路径
        file_path = os.path.join(directory, filename)

        # 构造命令
        command = [
            './target/debug/snowcap_main', 'synthesize', 'topology-zoo',
            file_path, 'FM2RR'
        ]
        print(f"Executing command: {' '.join(command)}")
        # 执行命令并捕获输出
        # 在这里使用 universal_newlines 来兼容 Python 2 和 3
        process = subprocess.Popen(command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
        stdout, stderr = process.communicate()  # 获取标准输出和标准错误
        print(stdout)

        output = stdout.strip()  # 获取标准输出的结果
        # 检查是否包含 'timeout' 字样
        if 'timeout' in output.lower():
            runtime = '-1'  # 如果出现 timeout，超时，记录 -1
        elif 'skipping extraction' in output.lower():
            runtime = '-2'  # 如果出现 skipping extraction，无解，记录 -2
        else:
            # 查找包含 '代码运行时间:' 的行，并提取后面的数字
            match = re.search(r'代码运行时间:\s*(.*)', output)

            if match:
                runtime = match.group(1)  # 获取运行时间数字
            else:
                runtime = '0'  # 如果没有找到这行，存储 0

        # 将结果写入输出文件
        with open(output_file, 'a') as f:
            f.write('{}\t{}\n'.format(filename, runtime))

        print('Executed {} successfully. Runtime: {}'.format(
            filename, runtime))

print("Experiment finished. Results saved to:", output_file)
