# -*- coding: utf-8 -*-
import subprocess
import re
import os
import sys
import select

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

        # 启动子进程
        process = subprocess.Popen(command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    bufsize=1)  # 行缓冲模式

        # 使用 select 来非阻塞地读取输出
        stdout_lines = []
        stderr_lines = []
        while True:
            # 检查文件描述符，使用 select 进行非阻塞读取
            rlist, _, _ = select.select([process.stdout, process.stderr], [], [], 1)
            
            if process.stdout in rlist:
                line = process.stdout.readline()
                if line:
                    stdout_lines.append(line)
                else:
                    # 如果 stdout 读取完毕
                    break  # 跳出循环，等待其他输出

            if process.stderr in rlist:
                line = process.stderr.readline()
                if line:
                    stderr_lines.append(line)
                    if 'timeout' in line.lower():  # 如果 stderr 中有 timeout，终止进程
                        print(f"Timeout detected, terminating process for {filename}")
                        process.terminate()
                        break  # 跳出循环
                else:
                    # 如果 stderr 读取完毕
                    break  # 跳出循环，等待其他输出

            # 如果进程结束，退出
            if process.poll() is not None:
                break

        # 确保进程已完全结束
        process.wait()

        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)

        print(stdout)  # 打印标准输出
        if stderr:
            print(stderr)  # 打印错误输出

        # 检查是否包含 'timeout' 字样
        if 'timeout' in stdout.lower() or 'timeout' in stderr.lower():
            runtime = '-1'  # 如果出现 timeout，超时，记录 -1
        elif 'skipping extraction' in stdout.lower():
            runtime = '-2'  # 如果出现 skipping extraction，无解，记录 -2
        else:
            # 查找包含 '代码运行时间:' 的行，并提取后面的数字
            match = re.search(r'代码运行时间:\s*(.*)', stdout)

            if match:
                runtime = match.group(1)  # 获取运行时间数字
            else:
                runtime = '0'  # 如果没有找到这行，存储 0

        # 将结果写入输出文件
        with open(output_file, 'a') as f:
            f.write('{}\t{}\n'.format(filename, runtime))

        print(f'Executed {filename} successfully. Runtime: {runtime}')

print("Experiment finished. Results saved to:", output_file)
