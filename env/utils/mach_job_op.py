import numpy as np
from itertools import accumulate
MAX = 1e6

AVAILABLE = 0 # 操作/机器当前可被调度
PROCESSED = 1 # 正在处理中（对操作）或忙（对机器）
COMPLETED = 3 # 已完成
FUTURE = 2    # 尚未到达可用时间（例如前序操作未完成）

class Machine:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.processed_op_history = []  # 记录在这台机器上已处理的所有操作信息
    
    def process_op(self, op_info):
        """
        调度一个操作到这台机器上。
        实际开始时间 = max(操作就绪时间, 机器空闲时间) → 体现“资源约束”。
        更新 op_info 的 start_time，并记录到历史。
        返回该操作的完成时间（用于更新后续操作的 avai_time）。
        """
        machine_avai_time = self.avai_time()
        start_time = max(op_info["current_time"], machine_avai_time)
        op_info["start_time"] = start_time
        finished_time = start_time + op_info["process_time"]
        self.processed_op_history.append(op_info)
        return finished_time


    def avai_time(self):
        """
        返回该机器下次可以使用的时刻。
        如果没有处理过任何操作，可用时间为 0。
        否则为最后一个操作的 开始时间 + 加工时间 = 完成时间。
        """
        if len(self.processed_op_history) == 0:
            return 0
        else:
            return self.processed_op_history[-1]["start_time"] + self.processed_op_history[-1]["process_time"]

    def get_status(self, current_time):
        """
        判断机器在 current_time 是否空闲。
        如果当前时间 ≥ 可用时间 → 空闲（AVAILABLE），否则忙碌（PROCESSED）。
        """
        if current_time >= self.avai_time():
            return AVAILABLE
        else:
            return PROCESSED

class Job:
    def __init__(self, args, job_id, op_config):
        self.args = args
        self.job_id = job_id
        self.operations = [Operation(self.args, self.job_id, config) for config in op_config]
        self.op_num = len(op_config)

        self.current_op_id = 0 # 当前待处理的操作索引（从0开始）
        # 计算从当前操作到结束的累计期望加工时间（用于启发式规则，如MWKR）
        self.acc_expected_process_time = list(accumulate([op.expected_process_time for op in self.operations[::-1]]))[::-1]

        
    def current_op(self):
        #返回当前待处理的操作对象（如果已完成所有操作，返回 None）。
        if self.current_op_id == -1:
            return None
        else:
            return self.operations[self.current_op_id]
    
    def update_current_op(self, avai_time):
        #更新当前操作的 avai_time（通常由前一个操作的完成时间决定）。
        self.operations[self.current_op_id].avai_time = avai_time 
    
    def next_op(self):
        #将指针移到下一个操作；如果已到最后一个，则设为 -1（表示作业完成）。
        if self.current_op_id + 1 < self.op_num:
            self.current_op_id += 1
        else:
            self.current_op_id = -1
        return self.current_op_id
    
    def done(self):
        #判断作业是否已完成（current_op_id == -1）。
        if self.current_op_id == -1:
            return True
        else:
            return False

class Operation:
    def __init__(self, args, job_id, config):
        self.args = args
        self.job_id = job_id
        self.op_id = config['id']
        self.machine_and_processtime = config['machine_and_processtime']  # [(machine_id, time), ...]
        self.node_id = -1# 可能用于图神经网络中的节点ID（预留）

        # 第一个操作（op_id=0）初始可用时间为0，其余设为MAX（等待前序完成）
        if self.op_id == 0:
            self.avai_time = 0
        else:
            self.avai_time = MAX

        self.start_time = -1  # -1 表示未调度
        self.finish_time = -1

        # 计算平均加工时间（用于启发式估计）
        total = 0
        for pair in self.machine_and_processtime:
            total += pair[1]
        self.expected_process_time = total / len(self.machine_and_processtime)
        
    def update(self, start_time, process_time):
        #当操作被调度后，记录实际开始和结束时间。
        self.start_time = start_time
        self.finish_time = start_time + process_time
    
    def get_status(self, current_time):
        #根据当前时间判断操作状态：
        """
        条件	状态
        未调度 (start_time == -1) 且 current_time >= avai_time	AVAILABLE（可调度）
        未调度 且 current_time < avai_time	FUTURE（还不能调度）
        已调度 且 current_time >= finish_time	COMPLETED
        已调度 且 current_time < finish_time	PROCESSED（正在加工）

        """
        if self.start_time == -1:
            if current_time >= self.avai_time:
                return AVAILABLE
            else:
                return FUTURE
        else:
            if current_time >= self.finish_time:
                return COMPLETED
            else:
                return PROCESSED