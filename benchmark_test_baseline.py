import os
from prompt import *
from benchmark.run_BoxNet1_baseline_methods import run_boxnet1_baselines
from benchmark.run_BoxLift_baseline_methods import run_boxlift_baselines
from benchmark.run_Blocksworld_baseline_methods import run_blocksworld_baselines
from benchmark.run_Boxnet_v2_baseline_methods import run_boxnet1_v2_baselines
from benchmark.run_Gridworld_baseline_methods import run_gridworld_baselines

if __name__ == '__main__':
    # gpt-4o, gpt-4o-mini, gpt-3.5-turbo for OpenAi API

    def log_run_info(log_file, run_info):
        with open(log_file, 'a') as f:
            f.write(run_info + "\n")

    model_name = "gpt-4o"  # gpt-4o, claude-3-5-sonnet-20241022, mistral-large-latest, open-mixtral-8x7b

    args_path = ''
    gather_save_input_dir = 'results_gather'
    for baseline_method_name in ['1_only_ques', 'code_interpreter', 'AutoGen', 'All_code_CoT', 'All_text']:

        save_input_dir = 'results_gather/BoxLift'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        run_boxlift_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

        save_input_dir = 'results_gather/BoxNet1_v2'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        run_boxnet1_v2_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

        save_input_dir = 'results_gather/blocksworld'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        run_blocksworld_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

        save_input_dir = 'results_gather/GridWorld1'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        run_gridworld_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)

        save_input_dir = 'results_gather/BoxNet1'
        if not os.path.exists(save_input_dir):
            os.makedirs(save_input_dir)
        run_boxnet1_baselines(save_input_dir, gather_save_input_dir, model_name, baseline_method_name, args_path)