"""
随机生成单个jsp或者fjsp中某个作业的实列
"""
import numpy as np
import random
import os
import time

#JSP
def gen_operations_JSP(op_num, machine_num, op_process_time_range): 
    op = []
    # 创建机器 ID 列表 [0, 1, ..., machine_num-1]
    m_seq = [i for i in range(machine_num)]
    # 随机打乱机器顺序（确保每个操作分配到不同机器，避免冲突）
    random.shuffle(m_seq)
    for op_id in range(op_num):
        # 随机生成该操作的加工时间
        process_time = np.random.randint(*op_process_time_range)
        # 将该操作绑定到打乱后的第 op_id 台机器
        mach_ptime = [(m_seq[op_id], process_time)]
        op.append({"id": op_id, "machine_and_processtime": mach_ptime})
    return op

# FJSP
def gen_operations_FJSP(machine_num, op_process_time_range): 
    op = []
    # 随机决定这个作业包含多少个操作（在 0.8~1.2 倍机器数之间）
    op_num = random.randint(int(0.8 * machine_num), int(1.2 * machine_num))

    for op_id in range(op_num):
        # 随机选择该操作可用的机器数量（至少1台，最多全部机器）
        random_size = np.random.choice(range(1, machine_num + 1, 1)) # the number of usable machine for this operation
        # 从 machine_num 台机器中无放回地随机选 random_size 台
        m_id = sorted(np.random.choice(machine_num, size=random_size, replace=False)) # the set of index of usable machine id with size random_size
        mach_ptime = []
        for id in m_id:
            process_time = np.random.randint(*op_process_time_range)
            mach_ptime.append((id, process_time))
        op.append({"id": op_id, "machine_and_processtime": mach_ptime})
    return op

if __name__ == '__main__':
    job_ops = gen_operations_FJSP(
        machine_num=6,
        op_process_time_range=(10, 80)
    )

    print(job_ops)


