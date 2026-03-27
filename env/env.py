import gym
import copy
import numpy as np
import json
import argparse
from env.utils.instance import JSP_Instance
from env.utils.mach_job_op import *
from env.utils.graph import Graph
from model.dpw_module import get_dynamic_priority_window
# Removed circular import: from model.gp_module import GPPriorityRuleEvolver, GPRuleCalculator

class JSP_Env(gym.Env):
    def __init__(self, args):
        self.args = args
        self.jsp_instance = JSP_Instance(args)
        self.gp_rule_calculator = None  # For online GP-DPW
        self.offline_rules = None  # For offline rules
        self.all_window_masks = []  # Pre-computed window masks for offline rules
        
        # Load offline rules if specified
        if getattr(args, 'use_offline_rules', False):
            self.load_offline_rules(args.rules_file)

    def step(self, action):
        if getattr(self.args, 'use_offline_rules', False):
            # Action is now a window index (0, 1, 2, 3, ...)
            if 0 <= action < len(self.all_window_masks):
                window_mask = self.all_window_masks[action]
                # Find the first available operation in the selected window
                avai_ops = self.jsp_instance.current_avai_ops()
                selected_op = None
                for i, op in enumerate(avai_ops):
                    if window_mask[i]:
                        selected_op = op
                        break
                
                if selected_op is None:
                    # Fallback: select first available operation
                    selected_op = avai_ops[0] if avai_ops else None
            else:
                # Invalid action, fallback
                avai_ops = self.jsp_instance.current_avai_ops()
                selected_op = avai_ops[0] if avai_ops else None
        else:
            # Original behavior: action is operation index
            avai_ops = self.jsp_instance.current_avai_ops()
            selected_op = avai_ops[action] if 0 <= action < len(avai_ops) else avai_ops[0]
        
        if selected_op is None:
            raise ValueError("No operation available for scheduling")
            
        current_makespan = self.get_makespan()
        self.jsp_instance.assign(selected_op)
        avai_ops = self.jsp_instance.current_avai_ops()
        next_makespan = self.get_makespan()
        return avai_ops, current_makespan - next_makespan, self.done()
    
    def reset(self):
        self.jsp_instance.reset()
        avai_ops = self.jsp_instance.current_avai_ops()
        
        # Pre-compute window masks if using offline rules
        if getattr(self.args, 'use_offline_rules', False) and self.offline_rules:
            self.compute_all_windows(avai_ops)
        
        return avai_ops
       
    def done(self):
        return self.jsp_instance.done()

    def get_makespan(self):
        return max(m.avai_time() for m in self.jsp_instance.machines)    
    
    def get_graph_data(self):
        return self.jsp_instance.get_graph_data()
        
    def load_instance(self, filename):
        self.jsp_instance.load_instance(filename)
        avai_ops = self.jsp_instance.current_avai_ops()
        
        # Pre-compute window masks if using offline rules
        if getattr(self.args, 'use_offline_rules', False) and self.offline_rules:
            self.compute_all_windows(avai_ops)
        
        return avai_ops
    
    def set_gp_rule_calculator(self, gp_calculator):
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
    
    def load_offline_rules(self, rules_file_path):
        """Load offline-generated GP rules from JSON file."""
        try:
            with open(rules_file_path, 'r') as f:
                self.offline_rules = json.load(f)
            print(f"Loaded {len(self.offline_rules)} offline rules from {rules_file_path}")
        except FileNotFoundError:
            print(f"Warning: Rules file {rules_file_path} not found. Using fallback behavior.")
            self.offline_rules = None
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {rules_file_path}. Using fallback behavior.")
            self.offline_rules = None
    
    def compute_all_windows(self, avai_ops):
        """Pre-compute priority windows for all offline rules."""
        if not self.offline_rules:
            self.all_window_masks = []
            return
        
        self.all_window_masks = []
        num_rules = getattr(self.args, 'num_rules', len(self.offline_rules))
        
        # Import here to avoid circular import
        from model.gp_module import GPPriorityRuleEvolver
        
        # Create a temporary GP evolver to compile rules
        temp_evolver = GPPriorityRuleEvolver()
        
        for i in range(num_rules):
            rule_key = f"rule_{i}"
            if rule_key not in self.offline_rules:
                # Fallback: use all operations
                mask = np.ones(len(avai_ops), dtype=bool)
            else:
                rule_str = self.offline_rules[rule_key]
                try:
                    # Parse the rule string back to a tree (simplified approach)
                    # In practice, you might want to store compiled rules
                    priorities = self._compute_priorities_from_rule_str(rule_str, avai_ops)
                    
                    # Select top operations based on priorities
                    window_size = getattr(self.args, 'dpw_window_size', 3)
                    if len(priorities) > window_size:
                        top_indices = np.argsort(-priorities)[:window_size]
                        mask = np.zeros(len(avai_ops), dtype=bool)
                        mask[top_indices] = True
                    else:
                        mask = np.ones(len(avai_ops), dtype=bool)
                        
                except Exception as e:
                    print(f"Error computing priorities for rule {i}: {e}")
                    mask = np.ones(len(avai_ops), dtype=bool)
            
            self.all_window_masks.append(mask)
    
    def _compute_priorities_from_rule_str(self, rule_str, avai_ops):
        """Compute priorities from a rule string (simplified implementation)."""
        # This is a placeholder - in practice, you'd need to properly parse
        # the GP tree string back to a callable function
        # For now, return random priorities as fallback
        return np.random.rand(len(avai_ops))
    
    def get_state_with_windows(self):
        """Get state representation including all window information."""
        if not getattr(self.args, 'use_offline_rules', False) or not self.all_window_masks:
            return None
        
        # Convert window masks to a tensor representation
        # Shape: (num_rules, num_operations)
        window_tensor = np.array(self.all_window_masks, dtype=float)
        return window_tensor
    