# Residual Scheduling: A New Reinforcement Learning Approach to Solving Job Shop Scheduling Problem

## Installation

Setup the virtual environment.
```c
podman run -it --name={YOUR_NAME}   -v $PWD/ResidualScheduling:/ResidualScheduling pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
```

Install required packages in the environment.
```c
pip install -r requirements.txt
```

Or install packages manually:
```c
pip install torch-geometric==2.3.1 torch opencv-python plotly matplotlib gym tensorboard pandas colorhash deap numpy scipy
```
## Run training
Follow the example to run a FJSP training procedure(RS). And there are some parameters for ablation studying.
```c
python3 train.py --date=train --instance_type=FJSP --data_size=10 --delete_node=true
```

## Reproduced the result in paper
Follow the example to run a FJSP testing 
```c
python3 test.py --date=test --instance_type=FJSP --delete_node=true --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS_FJSP/best'
```
Follow the example to run a FJSP testing (RS+op)
```c
python3 test.py --date=test --instance_type=FJSP --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS+op_FJSP/best'
```

### Similarly, for JSP
Follow the example to run a JSP testing (RS)
```c
python3 test.py --date=test --instance_type=JSP --delete_node=true --test_dir='./datasets/JSP/public_benchmark/ta' --load_weight='./weight/RS_JSP/best'
```
Follow the example to run a JSP testing (RS+op)
```c
python3 test.py --date=test --instance_type=JSP --test_dir='./datasets/JSP/public_benchmark/ta' --load_weight='./weight/RS+op_JSP/best'
```

## Hyperparameters list
```c
    python3 train.py \
    --device='cuda' \
    --instance_type='FJSP' \
    --data_size=10 \
    --max_process_time=100 \
    --delete_node=False \
    --entropy_coef=1e-2 \
    --episode=300001 \
    --lr=1e-4 \
    --step_size=1000 \
    --hidden_dim=256 \
    --GNN_num_layers=3 \
    --policy_num_layers=2 \
    --date='Dummy' \
    --detail=None \
    --test_dir='./datasets/FJSP/Brandimarte_Data' \
    --load_weight='./weight/RS_FJSP/best'
```

## Dynamic Priority Window (DPW) Feature

This project includes an enhanced **Dynamic Priority Window (DPW)** mechanism to further improve the makespan (total completion time) for FJSP problems. The DPW mechanism dynamically filters high-priority operations before scheduling, helping the model focus on decisions with the greatest impact on the global objective.

### How DPW Works

The DPW mechanism:
1. **Analyzes current system state** - Considers job criticality, operation urgency, processing times, and machine loads
2. **Selects top-priority operations** - Filters available operations to a smaller, more critical subset
3. **Guides decision-making** - The scheduling model selects only from this priority window
4. **Improves makespan** - By focusing on high-impact decisions, the overall schedule quality improves

### Using DPW in Training

To enable DPW during training with a window size of 3 operations:
```bash
python3 train.py --date=train --instance_type=FJSP --data_size=10 --use_dpw=True --dpw_window_size=3
```

### Using DPW in Testing

To test with DPW enabled:
```bash
python3 test.py --date=test --instance_type=FJSP --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS_FJSP/best' --use_dpw=True --dpw_window_size=3
```

### DPW Parameters

- `--use_dpw` (bool, default=False): Enable/disable the Dynamic Priority Window mechanism
- `--dpw_window_size` (int, default=3): Number of top-priority operations to include in the window. Smaller values increase focus on critical operations; larger values provide more flexibility

### DPW Priority Calculation

The DPW calculator uses weighted heuristics:
- **Job Criticality (40%)**: Operations from jobs with fewer remaining operations are prioritized (closer to completion affects makespan more)
- **Operation Urgency (35%)**: Operations that can complete sooner (bottleneck operations) are prioritized
- **Processing Time (15%)**: Operations with longer processing times have higher impact on makespan
- **Machine Load (10%)**: Operations on heavily-loaded machines are prioritized for better load balancing

### Example: Comparing with and without DPW

**Without DPW (baseline):**
```bash
python3 test.py --date=test_baseline --instance_type=FJSP --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS_FJSP/best' --use_dpw=False
```

**With DPW:**
```bash
python3 test.py --date=test_dpw --instance_type=FJSP --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS_FJSP/best' --use_dpw=True --dpw_window_size=3
```

Compare the results in `test_baseline` and `test_dpw` directories to see the improvement in makespan.

## Genetic Programming Enhanced Dynamic Priority Window (GP-DPW) Feature

This project now includes an advanced **Genetic Programming Enhanced Dynamic Priority Window (GP-DPW)** system that evolves adaptive scheduling rules using evolutionary computation. This creates a sophisticated "GP generates high-level strategies, DRL executes specific actions" collaborative framework.

### How GP-DPW Works

The GP-DPW system:
1. **GP Evolution**: Uses genetic programming to evolve rule trees that compute operation priorities
2. **Adaptive Rules**: Rules automatically adapt to different problem instances and characteristics  
3. **Collaborative Learning**: GP evolves strategies while DRL learns optimal execution
4. **Dynamic Adaptation**: Rules evolve during training to improve scheduling performance

### GP Components

- **Primitive Set**: Functions (+, -, *, /, max, min, sqrt, log, exp) and terminals (operation features)
- **Evolution Operators**: Tournament selection, one-point crossover, uniform mutation
- **Fitness Function**: Makespan minimization (1/makespan for maximization)
- **Rule Representation**: Tree structures that compute priority scores

### Prerequisites

Install DEAP for genetic programming:
```bash
pip install deap
```

### Using GP-DPW in Training

To enable GP-DPW during training:
```bash
python3 train.py --date=train_gp_dpw --instance_type=FJSP --data_size=10 --use_gp_dpw=True --gp_population_size=100 --gp_generations=50 --gp_evolve_interval=1000
```

### Using GP-DPW in Testing

To test with evolved GP rules:
```bash
python3 test.py --date=test_gp_dpw --instance_type=FJSP --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS_FJSP/best' --use_gp_dpw=True --gp_rule_path='./weight/GP_RULES/best_rule.pkl'
```

### GP-DPW Parameters

- `--use_gp_dpw` (bool, default=False): Enable GP-evolved DPW rules
- `--gp_population_size` (int, default=100): GP population size
- `--gp_generations` (int, default=50): Generations per evolution cycle
- `--gp_crossover_rate` (float, default=0.7): GP crossover probability
- `--gp_mutation_rate` (float, default=0.2): GP mutation probability
- `--gp_max_tree_depth` (int, default=5): Maximum depth of GP rule trees
- `--gp_evolve_interval` (int, default=1000): Episodes between GP evolution cycles
- `--gp_rule_path` (str): Path to save/load evolved GP rules

### Training Workflow

1. **Initialization**: Create GP evolver and DRL policy network
2. **Co-evolution**: 
   - DRL learns from current GP rules
   - GP evolves new rules periodically using DRL performance data
3. **Rule Injection**: Best GP rules are injected into environment for DRL use
4. **Iterative Improvement**: Process continues with increasingly sophisticated rules

### Example: Complete GP-DPW Training

```bash
# Train with GP-DPW evolution
python3 train.py \
    --date=train_gp_dpw \
    --instance_type=FJSP \
    --data_size=10 \
    --use_gp_dpw=True \
    --gp_population_size=100 \
    --gp_generations=50 \
    --gp_evolve_interval=1000 \
    --gp_max_tree_depth=5 \
    --episode=10000
```

### Example: Testing with Evolved Rules

```bash
# Test using evolved GP rules
python3 test.py \
    --date=test_gp_dpw \
    --instance_type=FJSP \
    --test_dir='./datasets/FJSP/Brandimarte_Data' \
    --load_weight='./weight/RS_FJSP/best' \
    --use_gp_dpw=True \
    --gp_rule_path='./weight/GP_RULES/best_rule.pkl'
```

### Performance Comparison

Compare different approaches:

**Baseline (No DPW):**
```bash
python3 test.py --date=baseline --use_dpw=False --use_gp_dpw=False
```

**Static DPW:**
```bash
python3 test.py --date=static_dpw --use_dpw=True --use_gp_dpw=False
```

**GP-DPW:**
```bash
python3 test.py --date=gp_dpw --use_dpw=False --use_gp_dpw=True --gp_rule_path='./weight/GP_RULES/best_rule.pkl'
```

### GP Rule Features

The GP system uses these operation features as terminals:
- `process_time`: Operation processing time
- `remaining_ops`: Remaining operations in job
- `job_progress`: Job completion percentage
- `machine_load`: Current machine load
- `machine_util`: Machine utilization rate
- `earliest_start`: Earliest possible start time
- `slack_time`: Operation slack time
- `current_time`: Current system time
- System constants and mathematical operators

### Expected Benefits

- **Adaptive Rules**: Rules automatically adapt to problem characteristics
- **Superior Performance**: GP can discover complex scheduling heuristics
- **Scalability**: Works across different problem sizes and types
- **Interpretability**: Evolved rules can be analyzed and understood
- **Continuous Improvement**: Rules evolve throughout training

### Technical Notes

- GP evolution runs in parallel with DRL training
- Rule trees are saved/loaded using pickle serialization
- Fitness evaluation uses multiple environment instances
- Bloat control prevents excessively complex rules
- Fallback to static DPW if GP rules unavailable
