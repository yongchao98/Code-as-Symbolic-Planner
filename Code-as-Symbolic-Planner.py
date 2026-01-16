import os
from prompt import *

from benchmark.run_BoxLift_RoboSteer_test import run_boxlift
from benchmark.run_Blocksworld_RoboSteer_test import run_blocksworld
from benchmark.run_BoxNet1_RoboSteer_test import run_boxnet1
from benchmark.run_Boxnet_v2_RoboSteer_test import run_boxnet1_v2
from benchmark.run_Gridworld_RoboSteer_test import run_gridworld

if __name__ == '__main__':
    # gpt-4o, gpt-4o-mini, gpt-3.5-turbo for OpenAi API

    def log_run_info(log_file, run_info):
        with open(log_file, 'a') as f:
            f.write(run_info + "\n")

    model_name = 'gpt-4o'  # gpt-4o, gpt-3.5-turbo, claude-3-5-sonnet-20241022, claude-3-sonnet-20240229, o1, o1-mini, o1-preview, gpt-4o, DeepSeek-R1
    CodeSteer_LLM = 'gpt-4o' # gpt-4o, claude-3-5-sonnet-20241022, mistral-large-latest, open-mixtral-8x7b
    gather_save_input_dir = 'results_gather'
    args_path = ''

    dataset_input_dir = 'dataset_gather/BoxNet1_v2_dataset'
    save_input_dir = 'results_gather/BoxNet1_v2'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 2, args_path, CodeSteer_LLM)
    ]:
        run_boxnet1_v2(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path,
                      CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/Blocksworld_dataset'
    save_input_dir = 'results_gather/blocksworld'
    runtime_list = []
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 2, args_path, CodeSteer_LLM)
    ]:
        run_blocksworld(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/BoxLift_dataset'
    save_input_dir = 'results_gather/BoxLift'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 2, args_path, CodeSteer_LLM)
    ]:
        run_boxlift(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/BoxNet1_v2_dataset'
    save_input_dir = 'results_gather/BoxNet1_v2'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 2, args_path, CodeSteer_LLM)
    ]:
        run_boxnet1_v2(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path,
                      CodeSteer_LLM)

    dataset_input_dir = 'dataset_gather/GridWorld1_dataset'
    save_input_dir = 'results_gather/GridWorld1'
    if not os.path.exists(save_input_dir):
        os.makedirs(save_input_dir)
    for dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM in [
        (dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, 3, args_path, CodeSteer_LLM)
    ]:
        run_gridworld(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path,
                      CodeSteer_LLM)
