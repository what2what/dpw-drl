import numpy as np
import copy
import os
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
from model.gp_module import GPPriorityRuleEvolver, GPRuleCalculator
from heuristic import *
from torch.utils.tensorboard import SummaryWriter
import json
import time

MAX = float(1e6) #设置全局变量使用设置一个最大值代表makespan

OPT_BY_INSTANCE = {
    "mk01": 39,
    "mk02": 26,
    "mk03": 204,
    "mk04": 60,
    "mk05": 172,
    "mk06": 58,
    "mk07": 139,
    "mk08": 523,
    "mk09": 307,
    "mk10": 197,
}

def evaluate(episode):
    eval_dir = './datasets/FJSP/Brandimarte_Data'
    instances = sorted(os.listdir(eval_dir))
    if len(instances) == 0:
        print("Eval: no instances found in {}".format(eval_dir))
        return None

    total_gap_per = 0.0
    gap_count = 0
    results = []

    policy.eval()
    with torch.no_grad():
        for instance in instances:
            file = os.path.join(eval_dir, instance)
            avai_ops = env.load_instance(file)
            while True:
                data, op_unfinished = env.get_graph_data()
                action_idx, _ = policy(
                    avai_ops,
                    data,
                    op_unfinished,
                    env.jsp_instance.graph.max_process_time,
                    greedy=True,
                )
                avai_ops, _, done = env.step(avai_ops[action_idx])
                if done:
                    ms = env.get_makespan()
                    inst_key = os.path.splitext(instance)[0].lower()
                    opt = OPT_BY_INSTANCE.get(inst_key)
                    gap = None
                    if opt is not None:
                        gap = ms - opt
                        total_gap_per += gap/opt
                        gap_count += 1
                    results.append((file, ms, opt, gap))
                    policy.clear_memory()
                    if opt is None:
                        print("Eval instance : {} \t\t Makespan : {} \t\t OPT : N/A \t\t Gap : N/A".format(file, ms))
                    else:
                        print("Eval instance : {} \t\t Makespan : {} \t\t OPT : {} \t\t Gap : {}".format(file, ms, opt, gap))
                    break

    avg_gap = None
    if gap_count > 0:
        avg_gap = total_gap_per / gap_count
    if avg_gap is None:
        print("Eval Episode : {} \t\t Avg Gap (ms - OPT) : N/A".format(episode))
    else:
        print("Eval Episode : {} \t\t Avg Gap (ms - OPT) : {}".format(episode, avg_gap))

    with open("./result/{}/eval.txt".format(args.date), "a") as outfile:
        for file, ms, opt, gap in results:
            outfile.write("episode : {} \t\t instance : {} \t\t makespan : {} \t\t opt : {} \t\t gap : {}\n".format(
                episode, file, ms, opt, gap
            ))
        outfile.write("episode : {} \t\t avg_gap(ms-opt) : {}\n".format(episode, avg_gap))

    policy.train()
    return avg_gap

def train():
    print("start Training")
    best_valid_makespan = MAX

    # Initialize GP evolver if using GP-DPW
    gp_evolver = None
    if getattr(args, 'use_gp_dpw', False):
        print("Initializing GP evolver for DPW...")
        gp_evolver = GPPriorityRuleEvolver(
            population_size=args.gp_population_size,
            generations=args.gp_generations,
            crossover_rate=args.gp_crossover_rate,
            mutation_rate=args.gp_mutation_rate,
            max_tree_depth=args.gp_max_tree_depth
        )
        # Set args reference for GP evolver
        gp_evolver.args = args

    for episode in range(0, args.episode):
        #每训练1000轮保存一次模型
        if episode % 1000 == 0 and episode > 0:
            torch.save(policy.state_dict(), "./weight/{}/{}".format(args.date, episode))
            evaluate(episode)

        # GP evolution at specified intervals
        if gp_evolver and episode % args.gp_evolve_interval == 0 and episode > 0:
            print(f"Episode {episode}: Evolving GP rules...")
            # Create some environment instances for GP evaluation
            eval_envs = []
            for _ in range(5):  # Use 5 instances for GP fitness evaluation
                eval_env = JSP_Env(args)
                eval_env.reset()
                eval_envs.append(eval_env)
            
            # Evolve GP rules
            best_rule_tree = gp_evolver.evolve(eval_envs)
            
            # Create GP rule calculator and set it in environment
            gp_calculator = GPRuleCalculator(best_rule_tree, gp_evolver.toolbox)
            env.set_gp_rule_calculator(gp_calculator)
            
            # Save the best rule
            os.makedirs(os.path.dirname(args.gp_rule_path), exist_ok=True)
            gp_evolver.save_rule_tree(best_rule_tree, args.gp_rule_path)
            print(f"GP rule evolved and saved to {args.gp_rule_path}")

        action_probs = []
        # 每一轮开始，重置环境，生成新的 JSP 算例
        avai_ops = env.reset()
        while avai_ops is None:
            avai_ops = env.reset()

        # 计算当前算例如果用启发式规则(MWKR)做，总耗时是多少，这作为一个“参考分”，用来衡量我们的 AI 是比规则好还是差
        MWKR_ms = heuristic_makespan(copy.deepcopy(env), copy.deepcopy(avai_ops), args.rule)

        while True:
            MWKR_baseline = heuristic_makespan(copy.deepcopy(env), copy.deepcopy(avai_ops), args.rule)
            #这里利用启发式规则来估计 “从当前状态到结束还需要多少时间”（Cost-to-Go）。这个 baseline 将被传入 policy 中计算 Advantage。
            baseline = MWKR_baseline - env.get_makespan()
            # 获取图数据
            data, op_unfinished = env.get_graph_data()
            
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
            
            # 放入网络，得到选择的动作索引(action_idx)和概率(action_prob)
            # action_idx refers to index in filtered_avai_ops
            action_idx, action_prob = policy(filtered_avai_ops, data, op_unfinished, env.jsp_instance.graph.max_process_time)
            
            # Map action index back to original avai_ops
            original_action_idx = original_indices[action_idx]
            
            # 环境前进一步，返回下一步的可选工序、奖励、是否结束
            avai_ops, reward, done = env.step(avai_ops[original_action_idx])

            # 注意这里的负号！ -reward 的解释：在调度问题中，env.step 返回的通常是增量时间（Cost）。
            # 我们要最小化时间，等价于最大化负的时间。所以把 Cost 变成负数作为 Reward 存进去。
            policy.rewards.append(-reward)
            policy.baselines.append(baseline)
            action_probs.append(action_prob)
            
            if done:
                # 清空梯度
                optimizer.zero_grad()
                # 调用我们在上一个问题中分析过的函数，利用收集到的 rewards 和 probs 计算 loss
                loss, policy_loss, entropy_loss = policy.calculate_loss(args.device)
                # 反向传播求梯度
                loss.backward()
                 #每 10 个 episode 记录一次 Loss 和 Action Probability，方便在网页上查看训练曲线。
                if episode % 10 == 0:
                    writer.add_scalar("action prob", np.mean(action_probs),episode)
                    writer.add_scalar("loss", loss, episode)
                    writer.add_scalar("policy_loss", policy_loss, episode)
                    writer.add_scalar("entropy_loss", entropy_loss, episode)
                # 更新网络权重
                optimizer.step()
                # 更新学习率
                scheduler.step()

                #至关重要，清空本轮的 log_probs 和 rewards，防止显存爆炸和数据污染。
                policy.clear_memory()
                # AI 跑出来的总时间
                ms = env.get_makespan()
                #如果 improve > 0，说明 AI 战胜了启发式规则。
                #如果 improve < 0，说明 AI 还没学好，跑得比贪婪规则还慢。
                improve = MWKR_ms - ms
                print("Date : {} \t\t Episode : {} \t\tJob : {} \t\tMachine : {} \t\tPolicy : {} \t\tImprove: {} \t\t MWKR : {}".format(
                    args.date, episode, env.jsp_instance.job_num, env.jsp_instance.machine_num, 
                    ms, improve, MWKR_ms))
                break

if __name__ == '__main__':
    #获取命令行参数（比如学习率、迭代次数、GPU设备等）。
    args = get_args()
    print(args)



    #创建保存结果 (result) 和模型权重 (weight) 的文件夹
    os.makedirs('./result/{}/'.format(args.date), exist_ok=True)
    os.makedirs('./weight/{}/'.format(args.date), exist_ok=True)

    #把本次运行的参数保存下来，方便日后复盘。
    with open("./result/{}/args.json".format(args.date),"a") as outfile:
        json.dump(vars(args), outfile, indent=8)
    #创建 JSP 调度环境。
    env = JSP_Env(args)
    #实例化我们在前面讨论过的 GNN+MLP 策略网络。
    policy = REINFORCE(args).to(args.device)
    #使用 Adam 优化器。
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    #学习率衰减策略。step_size 和 gamma 决定了每过多少个 episode 学习率下降一次，这有助于训练后期模型收敛。
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.99)
    #TensorBoard 记录器，用于画图监控 loss 变化。
    writer = SummaryWriter(comment=args.date)

    train()
