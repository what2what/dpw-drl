"""
Dynamic Priority Window (DPW) Module
This module implements a dynamic priority window mechanism for flexible job shop scheduling (FJSP).
The DPW filters the set of available operations to focus on high-priority jobs/operations based on heuristic rules.
"""

import numpy as np
from typing import List, Dict, Tuple


class DynamicPriorityWindowCalculator:
    """
    Calculator for Dynamic Priority Window (DPW) mechanism.
    
    The DPW mechanism selects a subset of high-priority jobs/operations from the available operations
    based on heuristic rules. This guides the scheduling policy to focus on decisions that have
    the most significant impact on the global objective (makespan minimization).
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize the DPW calculator.
        
        Args:
            window_size: Number of top-priority jobs/operations to include in the window.
                        Default is 3.
        """
        self.window_size = max(1, window_size)  # Ensure window_size >= 1
    
    def calculate_priority_window(self, avai_ops: List[Dict], 
                                   jobs: List, 
                                   machines: List,
                                   current_time: float) -> Tuple[List[Dict], np.ndarray]:
        """
        Calculate the dynamic priority window based on current system state.
        
        Heuristic rules applied (in order of priority):
        1. Operations with earliest completion time (bottleneck operations)
        2. Operations from jobs with most remaining operations (long jobs)
        3. Operations with longest processing time (critical operations)
        
        Args:
            avai_ops: List of available operations. Each operation is a dict with keys:
                     - 'job_id': Job ID
                     - 'op_id': Operation ID within the job
                     - 'm_id': Machine ID assigned to this operation
                     - 'process_time': Processing time for this operation
                     - 'node_id': Node ID in the graph
            jobs: List of Job objects (for retrieving job state)
            machines: List of Machine objects (for retrieving machine state)
            current_time: Current system time
            
        Returns:
            Tuple of (prioritized_ops, priority_mask) where:
            - prioritized_ops: List of operations in the dynamic priority window
            - priority_mask: Binary mask of shape (len(avai_ops),) indicating which operations are in the window
        """
        if len(avai_ops) == 0:
            return avai_ops, np.array([])
        
        if len(avai_ops) <= self.window_size:
            # If fewer or equal operations than window size, return all with mask of ones
            mask = np.ones(len(avai_ops), dtype=bool)
            return avai_ops, mask
        
        # Calculate priority scores for each available operation
        priority_scores = self._compute_priority_scores(avai_ops, jobs, machines, current_time)
        
        # Get indices of top-priority operations
        top_indices = np.argsort(-priority_scores)[:self.window_size]
        top_indices = np.sort(top_indices)  # Keep original order
        
        # Create mask
        mask = np.zeros(len(avai_ops), dtype=bool)
        mask[top_indices] = True
        
        # Extract prioritized operations
        prioritized_ops = [avai_ops[idx] for idx in top_indices]
        
        return prioritized_ops, mask
    
    def _compute_priority_scores(self, avai_ops: List[Dict], 
                                 jobs: List, 
                                 machines: List,
                                 current_time: float) -> np.ndarray:
        """
        Internal method to compute priority scores for each available operation.
        
        Combines multiple heuristic metrics:
        - Earliest completion time of the job's remaining operations
        - Number of remaining operations in the job
        - Processing time of the current operation
        - Machine load at the target machine
        
        Args:
            avai_ops: List of available operations
            jobs: List of Job objects
            machines: List of Machine objects
            current_time: Current system time
            
        Returns:
            Array of priority scores for each operation (higher = higher priority)
        """
        scores = np.zeros(len(avai_ops))
        
        for i, op_info in enumerate(avai_ops):
            job_id = op_info['job_id']
            op_id = op_info['op_id']
            m_id = op_info['m_id']
            process_time = op_info['process_time']
            
            # Get job and machine objects
            if 0 <= job_id < len(jobs) and 0 <= m_id < len(machines):
                job = jobs[job_id]
                machine = machines[m_id]
                
                # Metric 1: Criticality - Operations from jobs with fewer remaining operations
                # are more critical (they're closer to completion, affecting makespan)
                remaining_ops = job.remaining_op_num() if hasattr(job, 'remaining_op_num') else 1
                criticality_score = 1.0 / (remaining_ops + 1.0)
                
                # Metric 2: Urgency - Operations that can complete sooner have higher urgency
                # This prioritizes bottleneck operations on heavily-loaded machines
                machine_load = machine.avai_time() if hasattr(machine, 'avai_time') else 0
                job_earliest_time = job.current_op().avai_time if hasattr(job, 'current_op') else 0
                earliest_completion = max(machine_load, job_earliest_time) + process_time
                # Invert so that earlier completion = higher score
                urgency_score = 1.0 / (earliest_completion + 1.0)
                
                # Metric 3: Processing time - Longer processing times have higher impact on makespan
                processing_weight = process_time / 100.0  # Normalize by typical max process time
                
                # Metric 4: Machine load - Operations on heavily-loaded machines are more critical
                machine_utilization = machine_load / (current_time + 1e-6)
                
                # Combine metrics with weights
                score = (
                    0.40 * criticality_score +    # Job criticality
                    0.35 * urgency_score +         # Operation urgency
                    0.15 * processing_weight +     # Processing time importance
                    0.10 * machine_utilization     # Machine load balance
                )
                
                scores[i] = score
            else:
                # Fallback: use process time as score if job/machine cannot be accessed
                scores[i] = process_time / 100.0
        
        return scores


def get_dynamic_priority_window(avai_ops: List[Dict], 
                                jobs: List, 
                                machines: List,
                                current_time: float,
                                window_size: int = 3,
                                use_dpw: bool = True) -> Tuple[List[Dict], np.ndarray]:
    """
    Convenience function to compute dynamic priority window.
    
    Args:
        avai_ops: List of available operations
        jobs: List of Job objects
        machines: List of Machine objects
        current_time: Current system time
        window_size: Size of the priority window
        use_dpw: Whether to apply DPW filtering (if False, returns all operations)
        
    Returns:
        Tuple of (filtered_ops, mask)
    """
    if not use_dpw or len(avai_ops) == 0:
        mask = np.ones(len(avai_ops), dtype=bool) if len(avai_ops) > 0 else np.array([])
        return avai_ops, mask
    
    calculator = DynamicPriorityWindowCalculator(window_size=window_size)
    return calculator.calculate_priority_window(avai_ops, jobs, machines, current_time)
