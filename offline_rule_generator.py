"""
Offline Rule Generator for GP-DPW
This script uses Genetic Programming to evolve high-quality scheduling rules offline,
then saves the best rules to a JSON file for use in DRL training/testing.
"""

import json
import os
import argparse
from model.gp_module import GPPriorityRuleEvolver
from env.env import JSP_Env


def parse_args():
    parser = argparse.ArgumentParser(description='Offline GP Rule Generator')
    parser.add_argument('--population_size', type=int, default=100, help='GP population size')
    parser.add_argument('--generations', type=int, default=100, help='Number of GP generations')
    parser.add_argument('--crossover_rate', type=float, default=0.7, help='GP crossover rate')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='GP mutation rate')
    parser.add_argument('--max_tree_depth', type=int, default=5, help='Maximum GP tree depth')
    parser.add_argument('--top_n', type=int, default=4, help='Number of top rules to save')
    parser.add_argument('--output_file', type=str, default='./weight/GP_RULES/best_rules.json', help='Output JSON file path')
    parser.add_argument('--eval_instances', type=int, default=10, help='Number of instances for fitness evaluation')
    parser.add_argument('--device', type=str, default='cpu', help='Device for evaluation')
    return parser.parse_args()


def main():
    args = parse_args()

    print("Initializing GP Rule Evolver...")
    gp_evolver = GPPriorityRuleEvolver(
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        max_tree_depth=args.max_tree_depth
    )
    
    # Set args for the evolver (needed for environment creation)
    env_args = argparse.Namespace(
        instance_type='FJSP',
        data_size=10,
        max_process_time=100,
        delete_node=True,
        device=args.device
    )
    gp_evolver.args = env_args

    # Create evaluation environments
    print(f"Creating {args.eval_instances} evaluation environments...")
    eval_envs = []
    for _ in range(args.eval_instances):
        env = JSP_Env(env_args)
        env.reset()
        eval_envs.append(env)

    print("Starting GP evolution...")
    best_rule_tree = gp_evolver.evolve(eval_envs)

    print("Evolution complete. Selecting top rules...")
    # Get the top N rules from the final population
    top_individuals = gp_evolver.toolbox.selBest(gp_evolver.population, args.top_n)

    # Convert rules to string representation
    rules_dict = {}
    for i, individual in enumerate(top_individuals):
        rule_str = str(individual)
        rules_dict[f"rule_{i}"] = rule_str
        print(f"Rule {i} (fitness: {individual.fitness.values[0]:.4f}): {rule_str}")

    # Save to JSON file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(rules_dict, f, indent=2)

    print(f"Saved {args.top_n} best rules to {args.output_file}")

    # Also save the best individual as pickle for potential future use
    best_pickle_path = args.output_file.replace('.json', '_best.pkl')
    gp_evolver.save_rule_tree(best_rule_tree, best_pickle_path)
    print(f"Saved best rule tree to {best_pickle_path}")


if __name__ == '__main__':
    main()