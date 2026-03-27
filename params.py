import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for RL_GNN_JSP')
    # args for normal setting
    parser.add_argument('--device', type=str, default='cuda')
    # args for env
    parser.add_argument('--instance_type', type=str, default='FJSP')
    parser.add_argument('--data_size', type=int, default=10)
    parser.add_argument('--max_process_time', type=int, default=100, help='Maximum Process Time of an Operation')
    parser.add_argument('--delete_node', action='store_true', default=True) #消融实验控制是否进行剪枝 ture就是进行残差调度 false就是不进行剪枝
    # args for RL
    parser.add_argument('--entropy_coef', type=float, default=1e-2)
    parser.add_argument('--episode', type=int, default=300001)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=float, default=1000)
    # args for policy network
    parser.add_argument('--hidden_dim', type=int, default=256) #256
    # args for GNN
    parser.add_argument('--GNN_num_layers', type=int, default=3)
    # args for policy
    parser.add_argument('--policy_num_layers', type=int, default=2)
    
    # args for nameing
    parser.add_argument('--date', type=str, default='Dummy')
    parser.add_argument('--detail', type=str, default="no")
    # args for structure
    parser.add_argument('--rule', type=str, default='MWKR')

    # args for val/test
    parser.add_argument('--test_dir', type=str, default='./datasets/FJSP/Brandimarte_Data')
    parser.add_argument('--load_weight', type=str, default='./weight/RS_FJSP/best')
    
    # args for Dynamic Priority Window (DPW)
    parser.add_argument('--use_dpw', action='store_true', help='Whether to enable Dynamic Priority Window mechanism')
    parser.add_argument('--dpw_window_size', type=int, default=3, help='Size of the dynamic priority window (number of high-priority operations to consider)')
    
    # args for Genetic Programming (GP) DPW
    parser.add_argument('--use_gp_dpw', action='store_true', help='Whether to use GP-evolved rules for DPW instead of static heuristics')
    parser.add_argument('--gp_population_size', type=int, default=100, help='Population size for GP evolution')
    parser.add_argument('--gp_generations', type=int, default=50, help='Number of generations for GP evolution')
    parser.add_argument('--gp_crossover_rate', type=float, default=0.7, help='Crossover rate for GP')
    parser.add_argument('--gp_mutation_rate', type=float, default=0.2, help='Mutation rate for GP')
    parser.add_argument('--gp_max_tree_depth', type=int, default=5, help='Maximum depth of GP rule trees')
    parser.add_argument('--gp_evolve_interval', type=int, default=1000, help='Interval (episodes) for GP evolution during training')
    parser.add_argument('--gp_rule_path', type=str, default='./weight/GP_RULES/best_rule.pkl', help='Path to save/load GP rule trees')
    
    # args for Offline Rules
    parser.add_argument('--use_offline_rules', action='store_true', help='Whether to use offline-generated GP rules')
    parser.add_argument('--rules_file', type=str, default='./weight/GP_RULES/best_rules.json', help='Path to JSON file containing offline rules')
    parser.add_argument('--num_rules', type=int, default=4, help='Number of offline rules to use')

    args = parser.parse_args()
    return args
