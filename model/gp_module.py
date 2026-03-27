"""
Genetic Programming (GP) Module for Dynamic Priority Window (DPW)
This module implements a GP-driven system to evolve adaptive scheduling rules for FJSP.
The GP evolves rule trees that determine operation priorities for the DPW mechanism.
"""

import random
import numpy as np
import copy
from typing import List, Dict, Tuple, Any, Callable
import operator
import math
from deap import base, creator, tools, gp
from env.env import JSP_Env


class GPPriorityRuleEvolver:
    """
    Genetic Programming system for evolving priority rules for FJSP scheduling.
    
    This class uses GP to evolve rule trees that compute priority scores for operations
    in the Dynamic Priority Window mechanism. The evolved rules replace static heuristics
    with adaptive, learned scheduling strategies.
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 50,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.2,
                 tournament_size: int = 3,
                 max_tree_depth: int = 5,
                 min_tree_depth: int = 1,
                 elite_size: int = 5):
        """
        Initialize the GP evolver.
        
        Args:
            population_size: Number of individuals in the population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection
            max_tree_depth: Maximum depth of rule trees
            min_tree_depth: Minimum depth of rule trees
            elite_size: Number of elite individuals to preserve
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.max_tree_depth = max_tree_depth
        self.min_tree_depth = min_tree_depth
        self.elite_size = elite_size
        
        # Define primitive set (functions and terminals)
        self._setup_primitive_set()
        
        # Setup DEAP toolbox
        self._setup_toolbox()
        
        # Initialize population
        self.population = None
        self.best_individual = None
        self.best_fitness = -float('inf')
        
        # Statistics tracking
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
    
    def _setup_primitive_set(self):
        """Setup the primitive set for GP (functions and terminals)."""
        # Create primitive set
        self.pset = gp.PrimitiveSet("MAIN", arity=0)
        
        # Add functions (operators)
        self.pset.addPrimitive(operator.add, 2, name="add")
        self.pset.addPrimitive(operator.sub, 2, name="sub")
        self.pset.addPrimitive(operator.mul, 2, name="mul")
        self.pset.addPrimitive(self._protected_div, 2, name="div")
        self.pset.addPrimitive(max, 2, name="max")
        self.pset.addPrimitive(min, 2, name="min")
        self.pset.addPrimitive(math.sqrt, 1, name="sqrt")
        self.pset.addPrimitive(math.log, 1, name="log")
        self.pset.addPrimitive(math.exp, 1, name="exp")
        
        # Add terminals (features from FJSP environment)
        # Operation-level features
        self.pset.addTerminal("process_time", "process_time")  # Processing time of operation
        self.pset.addTerminal("remaining_ops", "remaining_ops")  # Remaining operations in job
        self.pset.addTerminal("job_progress", "job_progress")  # Job completion percentage
        self.pset.addTerminal("machine_load", "machine_load")  # Current load of target machine
        self.pset.addTerminal("machine_util", "machine_util")  # Machine utilization rate
        self.pset.addTerminal("earliest_start", "earliest_start")  # Earliest possible start time
        self.pset.addTerminal("slack_time", "slack_time")  # Slack time (latest start - earliest start)
        
        # System-level features
        self.pset.addTerminal("current_time", "current_time")  # Current system time
        self.pset.addTerminal("total_jobs", "total_jobs")  # Total number of jobs
        self.pset.addTerminal("total_machines", "total_machines")  # Total number of machines
        
        # Constants
        for i in range(-10, 11):
            self.pset.addTerminal(float(i), f"const_{i}")
    
    def _protected_div(self, left, right):
        """Protected division to avoid division by zero."""
        try:
            return left / right if right != 0 else 1.0
        except ZeroDivisionError:
            return 1.0
    
    def _setup_toolbox(self):
        """Setup DEAP toolbox for GP operations."""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # Setup toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, 
                            min_=self.min_tree_depth, max_=self.max_tree_depth)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)
        
        # Bloat control
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_depth))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_depth))
    
    def _evaluate_individual(self, individual):
        """Evaluate fitness of an individual (wrapper for DEAP)."""
        return (self.evaluate_fitness(individual),)
    
    def evaluate_fitness(self, individual, env_instances: List[JSP_Env] = None, num_evaluations: int = 5) -> float:
        """
        Evaluate the fitness of a rule tree individual.
        
        Fitness is based on makespan performance when using the rule tree
        to guide scheduling decisions.
        
        Args:
            individual: GP individual (rule tree)
            env_instances: List of environment instances for evaluation
            num_evaluations: Number of evaluation runs
            
        Returns:
            Fitness score (higher is better)
        """
        if env_instances is None:
            # Create some default instances for evaluation
            env_instances = []
            for _ in range(num_evaluations):
                env = JSP_Env(self.args)  # Need to pass args
                env.reset()
                env_instances.append(env)
        
        total_score = 0.0
        
        for env in env_instances:
            # Compile the rule tree into a callable function
            rule_func = self.toolbox.compile(expr=individual)
            
            # Evaluate using the rule tree
            makespan = self._simulate_scheduling_with_rule(env, rule_func)
            
            # Fitness = 1/makespan (higher makespan = lower fitness)
            if makespan > 0:
                total_score += 1.0 / makespan
            else:
                total_score += 0.0  # Penalize invalid schedules
        
        return total_score / len(env_instances)
    
    def _simulate_scheduling_with_rule(self, env: JSP_Env, rule_func: Callable) -> float:
        """
        Simulate scheduling using a rule tree to guide decisions.
        
        Args:
            env: FJSP environment instance
            rule_func: Compiled rule tree function
            
        Returns:
            Makespan (total completion time)
        """
        # Reset environment
        avai_ops = env.reset()
        
        while avai_ops:
            # Use rule tree to compute priorities
            priorities = []
            for op in avai_ops:
                # Extract features for this operation
                features = self._extract_operation_features(env, op)
                
                # Evaluate rule tree with features
                try:
                    priority = rule_func(**features)
                    priorities.append(float(priority))
                except:
                    # If evaluation fails, assign neutral priority
                    priorities.append(0.0)
            
            # Select operation with highest priority
            if priorities:
                best_idx = np.argmax(priorities)
                selected_op = avai_ops[best_idx]
            else:
                selected_op = avai_ops[0]  # Fallback
            
            # Execute the selected operation
            avai_ops, _, done = env.step(selected_op)
            
            if done:
                break
        
        return env.get_makespan()
    
    def _extract_operation_features(self, env: JSP_Env, op: Dict) -> Dict[str, float]:
        """
        Extract features for an operation to use as terminals in GP.
        
        Args:
            env: FJSP environment
            op: Operation dictionary
            
        Returns:
            Dictionary of feature values
        """
        job_id = op['job_id']
        op_id = op['op_id']
        m_id = op['m_id']
        process_time = op['process_time']
        
        # Get job and machine info
        job = env.jsp_instance.jobs[job_id] if 0 <= job_id < len(env.jsp_instance.jobs) else None
        machine = env.jsp_instance.machines[m_id] if 0 <= m_id < len(env.jsp_instance.machines) else None
        
        features = {
            'process_time': float(process_time),
            'remaining_ops': float(job.remaining_op_num()) if job and hasattr(job, 'remaining_op_num') else 1.0,
            'job_progress': 1.0 - (float(job.remaining_op_num()) / len(job.operations)) if job else 0.5,
            'machine_load': float(machine.avai_time()) if machine and hasattr(machine, 'avai_time') else 0.0,
            'machine_util': float(machine.avai_time()) / (env.jsp_instance.current_time + 1e-6) if machine else 0.0,
            'earliest_start': max(env.jsp_instance.current_time, 
                                job.current_op().avai_time if job and hasattr(job, 'current_op') else 0),
            'slack_time': 10.0,  # Placeholder - would need more complex calculation
            'current_time': float(env.jsp_instance.current_time),
            'total_jobs': float(len(env.jsp_instance.jobs)),
            'total_machines': float(len(env.jsp_instance.machines))
        }
        
        return features
    
    def evolve(self, env_instances: List[JSP_Env] = None) -> gp.PrimitiveTree:
        """
        Execute the complete GP evolution process.
        
        Args:
            env_instances: Environment instances for fitness evaluation
            
        Returns:
            Best evolved individual (rule tree)
        """
        # Initialize population
        self.population = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
        
        # Track best individual
        self._update_best_individual()
        
        # Evolution loop
        for gen in range(self.generations):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, len(self.population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population with offspring + elite
            elite = tools.selBest(self.population, self.elite_size)
            self.population[:] = elite + offspring
            
            # Update best individual
            self._update_best_individual()
            
            # Print statistics
            record = self.stats.compile(self.population)
            print(f"Generation {gen}: {record}")
        
        return self.best_individual
    
    def _update_best_individual(self):
        """Update the best individual found so far."""
        current_best = tools.selBest(self.population, 1)[0]
        if current_best.fitness.values[0] > self.best_fitness:
            self.best_individual = copy.deepcopy(current_best)
            self.best_fitness = current_best.fitness.values[0]
    
    def get_best_rule_tree(self) -> gp.PrimitiveTree:
        """Return the best evolved rule tree."""
        return self.best_individual
    
    def apply_rule_tree(self, rule_tree: gp.PrimitiveTree, env: JSP_Env, avai_ops: List[Dict]) -> np.ndarray:
        """
        Apply a rule tree to compute priority scores for operations.
        
        Args:
            rule_tree: Evolved GP individual
            env: Current environment state
            avai_ops: List of available operations
            
        Returns:
            Array of priority scores for each operation
        """
        rule_func = self.toolbox.compile(expr=rule_tree)
        priorities = []
        
        for op in avai_ops:
            features = self._extract_operation_features(env, op)
            try:
                priority = rule_func(**features)
                priorities.append(float(priority))
            except:
                priorities.append(0.0)  # Neutral priority on error
        
        return np.array(priorities)
    
    def save_rule_tree(self, rule_tree: gp.PrimitiveTree, filepath: str):
        """Save a rule tree to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(rule_tree, f)
    
    def load_rule_tree(self, filepath: str) -> gp.PrimitiveTree:
        """Load a rule tree from file."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class GPRuleCalculator:
    """
    Wrapper class for applying GP-evolved rule trees in the DPW system.
    """
    
    def __init__(self, rule_tree: gp.PrimitiveTree, toolbox):
        self.rule_tree = rule_tree
        self.toolbox = toolbox
        self.rule_func = toolbox.compile(expr=rule_tree)
    
    def compute_priorities(self, env: JSP_Env, avai_ops: List[Dict]) -> np.ndarray:
        """
        Compute priority scores for operations using the GP rule.
        
        Args:
            env: Current environment
            avai_ops: Available operations
            
        Returns:
            Priority scores array
        """
        priorities = []
        
        for op in avai_ops:
            # Extract features (simplified version)
            features = {
                'process_time': float(op['process_time']),
                'remaining_ops': 1.0,  # Placeholder
                'job_progress': 0.5,   # Placeholder
                'machine_load': 0.0,   # Placeholder
                'machine_util': 0.0,   # Placeholder
                'earliest_start': 0.0, # Placeholder
                'slack_time': 10.0,    # Placeholder
                'current_time': float(env.jsp_instance.current_time),
                'total_jobs': float(len(env.jsp_instance.jobs)),
                'total_machines': float(len(env.jsp_instance.machines))
            }
            
            try:
                priority = self.rule_func(**features)
                priorities.append(float(priority))
            except:
                priorities.append(0.0)
        
        return np.array(priorities)