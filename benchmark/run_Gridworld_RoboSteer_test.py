import json
import re
import pandas as pd
import os
import subprocess
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4, log_run_info
import random
import math
import json
from typing import List, Tuple, Dict
import time
import numpy as np
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_computational_approach, analyze_code_and_explain
import copy
import ast


#####################################################
# 1. Grid and Sample Generation
#####################################################

def generate_cells(rows: int, cols: int) -> List[str]:
    """
    Generate cell names as 'C{row},{col}' for a grid of given size.
    """
    cells = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            cells.append(f"C{r},{c}")
    return cells


def generate_adjacency(rows: int, cols: int) -> Dict[str, List[str]]:
    """
    Generate an adjacency dictionary for a grid.
    Two cells are adjacent if they share a side.
    """
    adjacency = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            cell = f"C{r},{c}"
            neighbors = []
            if c > 1:
                neighbors.append(f"C{r},{c - 1}")
            if c < cols:
                neighbors.append(f"C{r},{c + 1}")
            if r > 1:
                neighbors.append(f"C{r - 1},{c}")
            if r < rows:
                neighbors.append(f"C{r + 1},{c}")
            adjacency[cell] = neighbors
    return adjacency


def bfs_path_exists(start: str, end: str, adjacency: Dict[str, List[str]],
                    obstacles: List[str]) -> bool:
    """
    Simple BFS to check if there is a path from start to end
    without going through any obstacles.
    """
    if start == end:
        return True
    visited = set([start])
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current == end:
            return True
        for neighbor in adjacency[current]:
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def ensure_goals_reachable(initial: str, goals: List[str],
                           adjacency: Dict[str, List[str]],
                           obstacles: List[str]) -> bool:
    """
    Check if there's a sequence in which all goals can be visited starting from 'initial'.
    For simplicity, we check them in the given order (though the LLM can choose any order).
    This ensures at least one valid path exists.
    """
    current = initial
    for g in goals:
        if not bfs_path_exists(current, g, adjacency, obstacles):
            return False
        current = g
    return True


def synthesize_gridworld_sample(sample_id: str,
                                rows: int,
                                cols: int,
                                num_goals: int,
                                num_obstacles: int,
                                max_tries: int = 50) -> dict:
    """
    Create one Gridworld sample with:
      - a grid of size rows x cols
      - adjacency
      - random obstacles
      - random goals
      - a random initial robot position

    Ensures that at least one path exists to visit all goals.
    If it fails to find such a scenario in `max_tries`, it raises an error.
    """

    cells = generate_cells(rows, cols)
    adjacency = generate_adjacency(rows, cols)

    for _ in range(max_tries):
        # Randomly pick obstacles, goals, and initial position
        all_cells = cells[:]
        random.shuffle(all_cells)

        # First pick obstacles
        obstacles = all_cells[:num_obstacles]
        remaining = all_cells[num_obstacles:]

        # Then pick goals
        goals = remaining[:num_goals]
        remaining = remaining[num_goals:]

        # Pick initial from the leftover
        initial_pos = remaining[0]

        # Double-check that initial_pos is not an obstacle
        if initial_pos in obstacles:
            continue

        # Ensure there's at least one path that visits all goals in some order
        if ensure_goals_reachable(initial_pos, goals, adjacency, obstacles):
            sample = {
                "sample_id": sample_id,
                "grid": {
                    "rows": rows,
                    "cols": cols,
                    "cells": cells,
                    "adjacency": adjacency
                },
                "obstacles": obstacles,
                "goals": goals,
                "initial_robot": initial_pos,
                "description": (
                    f"This is a {rows}x{cols} gridworld. The robot starts at {initial_pos}, "
                    f"must visit all goals {goals}, and avoid obstacles {obstacles}."
                )
            }
            return sample

    raise ValueError("Failed to generate a solvable Gridworld sample after max_tries.")


def generate_gridworld_samples(num_samples: int = 5,
                               rows: int = 4,
                               cols: int = 4,
                               num_goals: int = 2,
                               num_obstacles: int = 4) -> List[dict]:
    """
    Generate multiple gridworld samples and save them as JSON files.
    """
    samples = []
    for i in range(num_samples):
        sample_id = f"gridworld_{i + 1}"
        sample = synthesize_gridworld_sample(sample_id, rows, cols,
                                             num_goals, num_obstacles)
        samples.append(sample)

        # Save each sample to a file
        filename = f"/Users/yongchaochen/Robot_NLP/RoboSteer/dataset_gather/GridWorld1_dataset/gridworld_sample_{rows}x{cols}_{i + 1}.json"
        with open(filename, "w") as f:
            json.dump(sample, f, indent=2)

        print(f"Generated sample {sample_id} -> {filename}")
    return samples


#####################################################
# 2. Prompt Generation
#####################################################

def generate_prompt(sample: dict) -> str:
    """
    Given a gridworld sample, generate a prompt instructing the LLM
    to output a valid path in JSON format enclosed in <<<>>>.
    """
    rows = sample['grid']['rows']
    cols = sample['grid']['cols']
    obstacles = sample['obstacles']
    goals = sample['goals']
    initial = sample['initial_robot']
    adjacency_str = json.dumps(sample['grid']['adjacency'], indent=2)

    prompt = f"""
You are given the following Gridworld planning problem:

Grid dimensions: {rows} x {cols}
Obstacles: {obstacles}
Goals: {goals}
Initial robot position: {initial}
Adjacency:
{adjacency_str}

Task:
- The robot must start at {initial}.
- The robot must visit all goals at least once (in any order).
- The robot must NOT pass through any obstacle cells.
- At each step, the robot can move to an adjacent cell (up, down, left, or right).
- Output your plan as a JSON list of robot positions (cells), from the initial position
  to the final position after all goals have been visited.
- The first position in your list must be the initial position.
- Enclose your final JSON list in <<< >>>. For example:
  <<<["C1,1", "C2,1", "C2,2", ...]>>>

Now provide your plan (a valid path):
"""
    return prompt.strip()


#####################################################
# 3. Extracting and Checking LLM Responses
#####################################################

def extract_json_from_response(response: str) -> str:
    """
    Extract the JSON string between <<< and >>> from the LLM response.
    Raises an error if not found.
    """
    pattern = r"<<<(.*?)>>>"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return '[]'
        raise ValueError("Could not find JSON enclosed by <<< and >>> in the response.")
    return matches[0].strip()

def extract_equation_with_GPT4(response):
    prompt = ('Your task is to extract the final answer from the given answer by another LLM:\n'
              'The final answer should be in the format <<<answer>>>, like <<<["C2,1","C3,1", ...]>>>.\n'
              'Return only the answer in that format.\n'
              'Input text: ')
    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                    user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation

def check_gridworld_plan_legality(plan: List[str],
                                  sample: dict) -> Tuple[bool, str]:
    """
    Check if the given path 'plan' is valid:
      1. The path starts at sample['initial_robot'].
      2. The path only moves in adjacency steps (up, down, left, right).
      3. The path never enters an obstacle.
      4. The path visits all goals at least once.
    Returns (True, "OK") if valid, or (False, "Error message") otherwise.
    """
    adjacency = sample['grid']['adjacency']
    obstacles = set(sample['obstacles'])
    goals = set(sample['goals'])
    initial = sample['initial_robot']

    # 1. Check start
    if not plan:
        return False, "Empty plan."
    if plan[0] != initial:
        return False, f"Plan does not start at initial position {initial}."

    # 2. Check adjacency and obstacle avoidance
    for i in range(len(plan) - 1):
        curr = plan[i]
        nxt = plan[i + 1]
        if nxt not in adjacency[curr]:
            return False, f"Invalid move from {curr} to {nxt} (not adjacent)."
        if nxt in obstacles:
            return False, f"Path enters an obstacle cell {nxt}."

    # 3. Check that all goals are visited at least once
    visited_goals = set(plan).intersection(goals)
    if visited_goals != goals:
        return False, f"Not all goals visited. Visited {visited_goals}, needed {goals}."

    return True, "OK"


def check_llm_response(response: str, sample: dict) -> Tuple[bool, str]:
    """
    1. Extract the JSON list from the LLM response.
    2. Parse it as a list of grid cells (strings).
    3. Check the plan for legality.
    """
    '''
    try:
        json_str = extract_json_from_response(response)
        plan = json.loads(json_str)
        print('1')
    except Exception as e:
        try:
            plan = json.loads(response)
            print('2')
        except Exception as e2:
            try:
                modify_response = extract_equation_with_GPT4(response)
                json_str = extract_json_from_response(modify_response)
                plan = json.loads(json_str)
                print('3')
            except:
                plan = []
    '''
    #try:
    #    json_str = extract_json_from_response(response)
    #except ValueError as e:
    #    return False, str(e)

    print('response:', response)
    try:
        # First try to parse the entire response
        plan = ast.literal_eval(response)
    except (ValueError, SyntaxError):
        # If that fails, extract the JSON part and try again
        json_str = extract_json_from_response(response)
        try:
            plan = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return False, "Extracted content is not valid JSON."

    if not isinstance(plan, list):
        return False, "The extracted JSON is not a list."
    if any(not isinstance(pos, str) for pos in plan):
        return False, "Some elements in the path list are not strings (cells)."

    is_valid, message = check_gridworld_plan_legality(plan, sample)
    return is_valid, message

def read_samples(filename="gridworld_sample_6x7_7.json"):
    """
    Read a Gridworld sample from the specified JSON file and return it as a Python dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    with open(filename, "r") as f:
        sample = json.load(f)

    return sample

def run_gridworld(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM):
    print('\n' + '*' * 30)
    print(f'Gridworld, Model_name: {model_name}, CodeSteer\n')
    base_save_code_dir = save_input_dir + f'/result_gridworld_{CodeSteer_LLM}_{model_name}_MTD_{max_tree_depth}_CodeSteer_1'

    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    total_sample_num = 0
    total_correct_num = 0
    grid_sizes = [(4, 4, 4, 5), (4, 5, 4, 5), (5, 5, 4, 5), (5, 5, 6, 8), (5, 6, 7, 10), (6, 6, 7, 10), (6, 7, 7, 10)]
    #grid_sizes = [(6, 6, 7, 10), (6, 7, 7, 10)]

    for rows, cols, num_goals, num_obstacles in grid_sizes:
        for iteration_num in range(5):
            total_sample_num += 1

            print('-------###-------###-------###-------')
            print(
                f'Row num is: {rows}, Column num is: {cols}, Iteration num is: {iteration_num}\n\n')

            save_code_dir = os.path.join(base_save_code_dir, f"gridworld_sample_{rows}x{cols}_{iteration_num + 1}/")
            if not os.path.exists(save_code_dir):
                os.makedirs(save_code_dir)

            sample = read_samples(dataset_input_dir + f'/gridworld_sample_{rows}x{cols}_{iteration_num + 1}.json')
            prompt = generate_prompt(sample)
            question = prompt

            response_list = [];
            CodeSteer_output_prompt_guidance_list = [];
            CodeSteer_input_prompt_list = [code_text_choice_prompt + question];
            CodeSteer_input_prompt_training_list = [code_text_choice_prompt + question]

            ############ Starting first guidance ############
            # starting_prompt_choice = GPT_response("", code_text_choice_prompt + question, model_name=model_name, code_interpreter=False,
            #                                    user_prompt_list=[code_text_choice_prompt + question], response_total_list=[], logprobs=False)

            starting_prompt_choice = with_COT_code_output_prompt

            print(f'Starting prompt choice: {starting_prompt_choice}')
            user_prompt_list = [starting_prompt_choice + question]
            CodeSteer_output_prompt_guidance_list.append(starting_prompt_choice)
            response = GPT_response('', user_prompt_list[0], model_name=model_name, code_interpreter=False,
                                    user_prompt_list=user_prompt_list, response_total_list=response_list,
                                    logprobs=False)
            response_list.append(response)

            ############ Further rounds of guidance ############
            for tree_depth in range(max_tree_depth):
                code_block_list = extract_code(response)
                if len(code_block_list) > 0:
                    code_complexity_summary, code_complexity_score = analyze_code_and_explain(code_block_list[0])
                    if code_complexity_score <= 2:
                        code_complexity_summary += '\nThe generated code may not be complex enough to carry out symbolic computing for solving the task.'
                    with open(save_code_dir + f"/code_1_{tree_depth}.py", "w") as f:
                        f.write(code_block_list[0])

                    try:
                        result = subprocess.run(
                            ["python3", save_code_dir + f"/code_1_{tree_depth}.py"],
                            capture_output=True, text=True, timeout=45
                        )
                        output = result.stdout
                        errors = result.stderr
                    except subprocess.TimeoutExpired as e:
                        output = e.stdout if e.stdout else ""
                        errors = e.stderr if e.stderr else ""
                        errors += f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"
                    #response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'

                    if isinstance(output, str):
                        if count_total_tokens([output + errors], []) > 8000:
                            response = response + f'\nThe execution result from the generated code is too long to be displayed.'
                        else:
                            response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'
                    else:
                        response = response + f'\nThe execution result from the generated code is:\nerrors: {errors}'

                check_code_saving_path = save_code_dir + f"/check_code_1_{tree_depth}.py"
                check_result = LLM_answer_code_checker(question, response, check_code_saving_path)

                CodeSteer_input_prompt_head = f'''{decision_prompt_complex_code} {question}\n'''
                if len(code_block_list) > 0:
                    print('\n############True#############\n')
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\nThe summary of generated code complexity is: {code_complexity_summary}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                else:
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                CodeSteer_input_prompt_total = CodeSteer_input_prompt_head + CodeSteer_input_prompt
                CodeSteer_input_prompt_list.append(CodeSteer_input_prompt_total)
                CodeSteer_input_prompt_training_list.append(CodeSteer_input_prompt)
                response_text = GPT_response("", '', model_name=model_name, code_interpreter=False,
                                             user_prompt_list=CodeSteer_input_prompt_list,
                                             response_total_list=CodeSteer_output_prompt_guidance_list,
                                             logprobs=False)
                matches = re.findall(r'<<<(.*?)>>>', response_text, re.DOTALL)
                guidance_prompt = matches[-1] if matches else response_text

                print(f'\nGuidance prompt_{tree_depth + 1}: {guidance_prompt}\n')

                CodeSteer_output_prompt_guidance_list.append(guidance_prompt)
                if '<<<Code>>>' in guidance_prompt:
                    guidance_prompt = with_COT_code_output_prompt
                elif '<<<Text>>>' in guidance_prompt:
                    guidance_prompt = text_output_prompt
                elif '<<<Return Answer>>>' in guidance_prompt or 'Return Answer' in guidance_prompt or '<<<Terminate>>>' in guidance_prompt or 'Terminate' in guidance_prompt:
                    break
                user_prompt_list.append(guidance_prompt)

                response = GPT_response('', user_prompt_list[0], model_name=model_name, code_interpreter=False,
                                        user_prompt_list=user_prompt_list, response_total_list=response_list,
                                        logprobs=False)

                response_list.append(response)
                # print(f'\nResponse_{tree_depth}: {response}\n')
            save_file_func(save_code_dir, response_list, user_prompt_list, question, CodeSteer_input_prompt_list,
                           CodeSteer_input_prompt_training_list, CodeSteer_output_prompt_guidance_list)

            ## Evaluation
            response = response_list[-1]
            original_response = response

            code_block_list = extract_code(response)
            for index, code_string in enumerate(code_block_list):
                with open(save_code_dir + f"/code_1_{index}.py", "w") as f:
                    f.write(code_string)
                # print(f'code_{index}:\n {code_string}')

            # Test the generated code
            if not os.path.exists(save_code_dir + f"/code_1_0.py"):
                pass
            else:
                try:
                    result = subprocess.run(
                        ["python3", "-c", f"exec(open('{save_code_dir}/code_1_0.py').read()); print(result)"],
                        capture_output=True,
                        text=True,
                        timeout=45
                    )

                    response = result.stdout
                    errors = result.stderr
                except Exception as e:
                    pass

            if count_total_tokens([response + errors], []) > 8000:
                response = 'The execution result from the generated code is too long to be displayed.'

            extracted_text_1, _ = extract_and_check(response)
            if extracted_text_1 == '':
                extracted_text_1 = extract_equation_with_GPT4(response)

            extracted_text_2, _ = extract_and_check(original_response)
            if extracted_text_2 == '':
                extracted_text_2 = extract_equation_with_GPT4(original_response)

            is_correct_1, message_1 = check_llm_response(response, sample)
            is_correct_2, message_2 = check_llm_response(original_response, sample)

            print(f"\nResponse 1: {extracted_text_1}")
            print(f"Response 2: {extracted_text_2}")
            print(f"Is correct 1: {is_correct_1}, message 1: {message_1}")
            print(f"Is correct 2: {is_correct_2}, message 2: {message_2}")

            with open(save_code_dir + "/response_answer_1.txt", "w") as f:
                f.write(extracted_text_1)

            with open(save_code_dir + "/response_answer_2.txt", "w") as f:
                f.write(extracted_text_2)

            if is_correct_1 == True or is_correct_2 == True:
                print('True')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('True')
                total_correct_num += 1
            else:
                print('False')
                with open(save_code_dir + f"/success_failure.txt", "w") as f:
                    f.write('False')

            print(f'\ntotal_sample_num: {total_sample_num}')
            print(f'total_correct_num: {total_correct_num}')
            print(f'Correct/all: {total_correct_num}/{total_sample_num}')

    with open(base_save_code_dir + f"/total_sample_num.txt", "w") as f:
        f.write(str(total_sample_num))
    with open(base_save_code_dir + f"/total_correct_num.txt", "w") as f:
        f.write(str(total_correct_num))

    run_info = f"CodeSteer, Gridworld, {CodeSteer_LLM}, {model_name}, MTD_{max_tree_depth}_CodeSteer_1\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)

#####################################################
# Example of generating a few Gridworld samples
'''
for rows, cols, num_goals, num_obstacles in [(4, 4, 4, 5), (4, 5, 4, 5), (5, 5, 4, 5), (5, 5, 6, 8), (5, 6, 7, 10), (6, 6, 7, 10), (6, 7, 7, 10)]:
    samples = generate_gridworld_samples(
        num_samples=20,  # how many samples
        rows=rows,  # grid rows
        cols=cols,  # grid cols
        num_goals=num_goals,  # how many goals
        num_obstacles=num_obstacles  # how many obstacles
    )

    # Show how to create a prompt and do a mock "response check".
    for i, sample in enumerate(samples):
        prompt = generate_prompt(sample)
        print(f"\n--- Sample {rows}, {cols}, {i + 1}\n")

        # Suppose we have a hypothetical LLM response (you would replace this with a real LLM call).
        # Here we produce a "fake" path that includes only cell names:
        # We'll move from the initial cell, do a couple steps, and then
        # include the first and last goal from sample['goals'].
        model_name = "gpt-4o"  # Example model name; adjust as needed.
        if model_name in ['o1', "o1-preview", 'o1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo',
                          "claude-3-sonnet-20240229",
                          "claude-3-opus-20240229", "claude-3-haiku-20240307"]:
            response = GPT_response('', prompt, model_name=model_name,
                                    code_interpreter=False, user_prompt_list=[prompt],
                                    response_total_list=[], logprobs=False)

        is_correct, message = check_llm_response(response, sample)
        print(f"LLM response correctness: {is_correct}, message: {message}")
        #print("LLM response:\n", response)
'''