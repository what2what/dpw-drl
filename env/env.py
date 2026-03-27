import gym
import copy
import numpy as np
from env.utils.instance import JSP_Instance
from env.utils.mach_job_op import *
from env.utils.graph import Graph
from model.dpw_module import get_dynamic_priority_window
from model.gp_module import GPRuleCalculator

class JSP_Env(gym.Env):
    def __init__(self, args):
        self.args = args
        self.jsp_instance = JSP_Instance(args)
        self.gp_rule_calculator = None  # Will be set by training/test scripts

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
    
    def set_gp_rule_calculator(self, gp_calculator: GPRuleCalculator):
        """Set the GP rule calculator for DPW."""
        self.gp_rule_calculator = gp_calculator
    
    def get_dynamic_priority_window_from_gp(self, avai_ops: list, window_size: int = 3):
        """
        Compute dynamic priority window using GP-evolved rules.
        
        This method uses a GP-evolved rule tree to compute priority scores
        for operations and select the top-priority ones for the DPW.
        
        Args:
            avai_ops: List of available operations to filter
            window_size: Number of top-priority operations to include in the window
            
        Returns:
            Tuple of (prioritized_ops, priority_mask) where:
            - prioritized_ops: Filtered list of high-priority operations
            - priority_mask: Binary mask indicating which operations are in the window
        """
        if self.gp_rule_calculator is None:
            # Fallback to original DPW if no GP rule is set
            return self.get_dynamic_priority_window(avai_ops, window_size)
        
        if len(avai_ops) == 0:
            return avai_ops, np.array([])
        
        if len(avai_ops) <= window_size:
            mask = np.ones(len(avai_ops), dtype=bool)
            return avai_ops, mask
        
        # Use GP rule to compute priorities
        priorities = self.gp_rule_calculator.compute_priorities(self, avai_ops)
        
        # Get indices of top-priority operations
        top_indices = np.argsort(-priorities)[:window_size]
        top_indices = np.sort(top_indices)
        
        # Create mask
        mask = np.zeros(len(avai_ops), dtype=bool)
        mask[top_indices] = True
        
        # Extract prioritized operations
        prioritized_ops = [avai_ops[idx] for idx in top_indices]
        
        return prioritized_ops, mask
    
    def get_dynamic_priority_window(self, avai_ops: list, window_size: int = 3):
        """
        Legacy method: Compute dynamic priority window using static heuristics.
        
        This is kept for backward compatibility when GP is not used.
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
    