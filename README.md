# Residual Scheduling: A New Reinforcement Learning Approach to Solving Job Shop Scheduling Problem

## Installation

Setup the virtual environment.
```c
podman run -it --name={YOUR_NAME}   -v $PWD/ResidualScheduling:/ResidualScheduling pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
```

Install required packages in the environment.
```c
pip install torch-geometric==2.3.1  opencv-python plotly matplotlib gym tensorboard pandas colorhash
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
