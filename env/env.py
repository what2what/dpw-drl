import gym
import copy
import numpy as np
from env.utils.instance import JSP_Instance
from env.utils.mach_job_op import *
from env.utils.graph import Graph
from model.dpw_module import get_dynamic_priority_window

class JSP_Env(gym.Env):
    def __init__(self, args):
        self.args = args
        self.jsp_instance = JSP_Instance(args)

    def step(self, step_op):
        current_makespan = self.get_makespan()
        self.jsp_instance.assign(step_op)
        avai_ops = self.jsp_instance.current_avai_ops()
        next_makespan = self.get_makespan()
        return avai_ops, current_makespan - next_makespan, self.done()
    
    def reset(self):
        self.jsp_instance.reset()
        return self.jsp_instance.current_avai_ops()
       
    def done(self):
        return self.jsp_instance.done()

    def get_makespan(self):
        return max(m.avai_time() for m in self.jsp_instance.machines)    
    
    def get_graph_data(self):
        return self.jsp_instance.get_graph_data()
        
    def load_instance(self, filename):
        self.jsp_instance.load_instance(filename)
        return self.jsp_instance.current_avai_ops()
    
    def get_dynamic_priority_window(self, avai_ops: list, window_size: int = 3):
        """
        Compute and return a dynamic priority window of high-priority operations.
        
        This method filters the available operations to focus on those with the highest priority
        based on heuristic rules considering:
        - Job criticality (remaining operations)
        - Operation urgency (earliest completion time)
        - Processing time importance
        - Machine load balance
        
        Args:
            avai_ops: List of available operations to filter
            window_size: Number of top-priority operations to include in the window
            
        Returns:
            Tuple of (prioritized_ops, priority_mask) where:
            - prioritized_ops: Filtered list of high-priority operations
            - priority_mask: Binary mask indicating which operations are in the window
        """
        use_dpw = getattr(self.args, 'use_dpw', False)
        if not use_dpw or len(avai_ops) == 0:
            mask = np.ones(len(avai_ops), dtype=bool) if len(avai_ops) > 0 else np.array([])
            return avai_ops, mask
        
        dpw_window_size = getattr(self.args, 'dpw_window_size', window_size)
        prioritized_ops, mask = get_dynamic_priority_window(
            avai_ops=avai_ops,
            jobs=self.jsp_instance.jobs,
            machines=self.jsp_instance.machines,
            current_time=self.jsp_instance.current_time,
            window_size=dpw_window_size,
            use_dpw=True
        )
        
        return prioritized_ops, mask
    