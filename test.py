import torch
import numpy as np
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
from model.gp_module import GPPriorityRuleEvolver, GPRuleCalculator
import time
import os

def test():
    for instance in os.listdir(args.test_dir):
        file = os.path.join(args.test_dir, instance)
        avai_ops = env.load_instance(file)
        st = time.time()

        while True:
            data, op_unfinished= env.get_graph_data()
            
            # Get window tensor if using offline rules
            window_tensor = env.get_state_with_windows()
            
            # Apply Dynamic Priority Window (DPW) filtering
            use_dpw = getattr(args, 'use_dpw', False)
            use_gp_dpw = getattr(args, 'use_gp_dpw', False)
            
            if use_gp_dpw and env.gp_rule_calculator is not None:
                # Use GP-evolved rules
                dpw_window_size = getattr(args, 'dpw_window_size', 3)
                filtered_avai_ops, priority_mask = env.get_dynamic_priority_window_from_gp(
                    avai_ops, 
                    window_size=dpw_window_size
                )
                original_indices = np.where(priority_mask)[0].tolist() if len(priority_mask) > 0 else list(range(len(avai_ops)))
            elif use_dpw:
                # Use static heuristics
                dpw_window_size = getattr(args, 'dpw_window_size', 3)
                filtered_avai_ops, priority_mask = env.get_dynamic_priority_window(
                    avai_ops, 
                    window_size=dpw_window_size
                )
                original_indices = np.where(priority_mask)[0].tolist() if len(priority_mask) > 0 else list(range(len(avai_ops)))
            else:
                # No DPW filtering
                filtered_avai_ops = avai_ops
                original_indices = list(range(len(avai_ops)))
            
            action_idx, action_prob = policy(filtered_avai_ops, data, op_unfinished, env.jsp_instance.graph.max_process_time, greedy=True, window_tensor=window_tensor)
            
            # Map action index back to original avai_ops if not using offline rules
            if not getattr(args, 'use_offline_rules', False):
                original_action_idx = original_indices[action_idx]
                selected_op = avai_ops[original_action_idx]
            else:
                # For offline rules, action_idx is already the window index, step() will handle it
                selected_op = action_idx
            
            avai_ops, _, done = env.step(selected_op)
            
            if done:
                ed = time.time()
                policy.clear_memory()

                print("instance : {}, ms : {}, time : {}".format(file, env.get_makespan(), ed - st))
                with open("./result/{}/test_result.txt".format(args.date),"a") as outfile:
                    outfile.write(f'instance : {file:60}, policy : {env.get_makespan():10}\t')
                    outfile.write(f'time : {ed - st:10}\n')
                break

if __name__ == '__main__':
    args = get_args()
    print(args)
    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)
    os.makedirs('./result/{}/'.format(args.date), exist_ok=True)
    
    policy.load_state_dict(torch.load(args.load_weight, map_location=args.device), False)
    
    # Load GP rule if using GP-DPW or offline rules
    use_gp_dpw = getattr(args, 'use_gp_dpw', False)
    use_offline_rules = getattr(args, 'use_offline_rules', False)
    
    if use_offline_rules:
        # Offline rules are loaded automatically in JSP_Env.__init__
        print("Using offline GP rules for window selection")
    elif use_gp_dpw:
        if os.path.exists(args.gp_rule_path):
            print(f"Loading GP rule from {args.gp_rule_path}")
            gp_evolver = GPPriorityRuleEvolver()  # Create instance to get toolbox
            gp_evolver.args = args
            best_rule_tree = gp_evolver.load_rule_tree(args.gp_rule_path)
            gp_calculator = GPRuleCalculator(best_rule_tree, gp_evolver.toolbox)
            env.set_gp_rule_calculator(gp_calculator)
            print("GP rule loaded and set in environment")
        else:
            print(f"Warning: GP rule path {args.gp_rule_path} not found. Falling back to static DPW.")
    
    with torch.no_grad():
        test()
                    