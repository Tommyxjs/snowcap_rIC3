# -*- coding: utf-8 -*-
import subprocess
import re
import os
import sys
import select

directory = './eval_sigcomm2021/topology_zoo/'
output_file = 'experiment_results_cdcl_3.txt'

# 确保输出文件是空的，或者创建新的
with open(output_file, 'w') as f:
    f.write('Filename\tLength\tType\tRuntime\n')

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
            print("Executing command: " + ' '.join(command))

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
                rlist, _, _ = select.select([process.stdout, process.stderr], [], [], 1)
                
                if process.stdout in rlist:
                    line = process.stdout.readline()
                    if line:
                        stdout_lines.append(line)
                    else:
                        break

                if process.stderr in rlist:
                    line = process.stderr.readline()
                    if line:
                        stderr_lines.append(line)
                        if 'timeout' in line.lower():
                            print("Timeout detected, terminating process for {}".format(filename))
                            process.terminate()
                            break
                    else:
                        break

                if process.poll() is not None:
                    break

            process.wait()

            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)

            print(stdout)
            if stderr:
                print(stderr)

            if 'timeout' in stdout.lower() or 'timeout' in stderr.lower():
                runtime = '-1'
            else:
                match = re.search(r'代码运行时间:\s*(.*)', stdout)
                length = re.search(r'self\.groups\.len\(\):\s*(\d+)', stdout)
                error_match = re.search(r'Error checking policies:\s*(\w+)', stdout)

                if error_match:
                    error_type_str = error_match.group(1)
                    if error_type_str == 'ForwardingLoops':
                        error_type = "2"
                    elif error_type_str == 'ForwardingBlackHole':
                        error_type = "1"
                    else:
                        error_type = "0"
                else:
                    error_type = "0"

                if length:
                    l = length.group(1)
                else:
                    l = '0'

                if match:
                    runtime = match.group(1)
                else:
                    runtime = '0'

            f.write('{}\t{}\t{}\t{}\n'.format(filename, l, error_type, runtime))

            print('Executed {} successfully. Length:{} and Runtime: {}'.format(filename, l, runtime))

print("Experiment finished. Results saved to: {}".format(output_file))
