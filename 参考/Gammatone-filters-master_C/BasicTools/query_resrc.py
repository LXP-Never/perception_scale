import sys
import subprocess
import numpy as np
import re
import argparse


def check_cpu(is_print=True):
    BASE = 1024

    # top -bn1
    cmd_line = ['top',"-bn2",'-d 0.1']
    info = subprocess.run(cmd_line,check=True,stdout=subprocess.PIPE).stdout

    #
    cpu_info = [line for line in info.decode().split('\n') if 'Cpu(s)' in line][-1]
    cpu_ratio = np.float(cpu_info.split()[1])

    #
    mem_info = [line for line in info.decode().split('\n') if 'MiB Mem' in line][-1]
    mem_info_elems = re.split(string=mem_info,pattern='[:,]')
    mem_all = np.float(mem_info_elems[1].split()[0])
    mem_used = np.float(mem_info_elems[2].split()[0])
    mem_ratio = mem_used/mem_all

    if is_print:
        print('cpu:{:.2f}%  mem:{:.2f}%'.format(cpu_ratio,mem_ratio))

    return [cpu_ratio,mem_ratio]


def check_gpu():
    print('check_gpu')
    cmd_line = ['nvidia-smi']
    info = subprocess.run(cmd_line,check=True,stdout=subprocess.PIPE).stdout.decode('utf8');

    n_gpu = info.count('GeForce')

    lines = info.split('\n')
    gpu_linenum_base = 8
    gpu_info = dict()
    for i in range(n_gpu):
        # mem_used,
        text = lines[gpu_linenum_base+i*3]
        _,mem_seg,_ = filter(lambda x:len(x)>0,text.split('|'))
        mem_used, mem_all = [int(elem.strip()[:-3])
                                    for elem in mem_seg.strip().split('/')]
        gpu_info[str(i)]={'mem_all':mem_all,
                          'mem_used':mem_used,
                          'mem_avaliable':mem_all-mem_used,
                          'usage':dict()}

    pid_info_text = subprocess.run(['ps','aux'],check=True,stdout=subprocess.PIPE).stdout.decode('utf8')
    # [print(i,line)  for i,line in enumerate(pid_info_text.split('\n')[1:-1])]
    pid_user_info = {line.split()[1]:line.split()[0] for line in pid_info_text.split('\n')[1:-1] }
    pid_linenum_base = gpu_linenum_base+(n_gpu-1)*3+7
    for line in lines[pid_linenum_base:-2]:
        gpu_id,pid,type,process_name,mem_used_text = line.strip('| \n').split()
        mem_used = int(mem_used_text[:-3])
        user_name = pid_user_info[pid]

        if user_name in gpu_info[gpu_id]['usage']:
            gpu_info[gpu_id]['usage'][user_name] = gpu_info[gpu_id]['usage'][user_name]+mem_used
        else:
            gpu_info[gpu_id]['usage'][user_name] = mem_used

    for gpu_id in gpu_info.keys():
        print(f'GPU_ID: {gpu_id}')
        print('-'*50)
        print('Memory all:{1[mem_all]} used:{1[mem_used]} avaliable:{1[mem_avaliable]}'.format(gpu_id, gpu_info[gpu_id]))
        usage_str = '       '
        for user_name in gpu_info[gpu_id]['usage'].keys():
            usage_str = usage_str + '{} {}; '.format(user_name, gpu_info[gpu_id]['usage'][user_name])
        print(usage_str)


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--pc', dest='pc', type=str,
                        default='localhost', help='username@ip, default to run on local machine')
    parser.add_argument('--device', dest='device', type=str,
                        default='cpu', help='cpu or gpu')
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    if args.pc == 'localhost':
        if args.device == 'cpu':
            check_cpu()
        elif args.device == 'gpu':
            check_gpu()
        else:
            print(args.device)
    else:
        
        cmd = ' '.join(['ssh', args.pc, 'python', '-m' ,'BasicTools.query_resrc', '--device', args.device])
        print(cmd)
        ssh = subprocess.Popen(['ssh', f'{args.pc}', f'source ~/.bashrc; python -m BasicTools.query_resrc --device {args.device}'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        result = ssh.stdout.readlines()
        print(result)

if __name__ == '__main__':
    main()
