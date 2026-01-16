import json
import random
import os
import re

from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, extract_code, extract_and_check, LLM_answer_code_checker, save_file_func, paraphrase_with_GPT4, log_run_info
import math

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
#from models.src.llamafactory.chat.chat_model import run_response
import copy
import ast

########################################
# 1. Sample Synthesis and Reading
########################################

def generate_cells(rows, cols):
    """Generate cell names as 'C{row},{col}' for a grid of given size."""
    cells = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            cells.append(f"C{r},{c}")
    return cells

def generate_adjacency(rows, cols):
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
                neighbors.append(f"C{r},{c-1}")
            if c < cols:
                neighbors.append(f"C{r},{c+1}")
            if r > 1:
                neighbors.append(f"C{r-1},{c}")
            if r < rows:
                neighbors.append(f"C{r+1},{c}")
            adjacency[cell] = neighbors
    return adjacency

def random_assignment(cells, num_boxes):
    """
    Randomly assign initial and goal positions for boxes.
    Returns a tuple (boxes, initial_state, goal_locations).
    """
    boxes = [f"box{i+1}" for i in range(num_boxes)]
    init_positions = random.sample(cells, num_boxes)
    goal_positions = random.sample(cells, num_boxes)
    initial_state = {box: pos for box, pos in zip(boxes, init_positions)}
    goal_locations = {box: pos for box, pos in zip(boxes, goal_positions)}
    return boxes, initial_state, goal_locations

def synthesize_sample(sample_id, rows, cols, num_boxes):
    """
    Synthesize one sample planning problem.
    Each cell has one robot arm.
    """
    cells = generate_cells(rows, cols)
    adjacency = generate_adjacency(rows, cols)
    arms = cells.copy()  # one arm per cell
    boxes, initial_state, goal_locations = random_assignment(cells, num_boxes)

    sample = {
        "sample_id": sample_id,
        "grid": {
            "rows": rows,
            "cols": cols,
            "cells": cells,
            "adjacency": adjacency
        },
        "arms": arms,
        "boxes": boxes,
        "initial_state": initial_state,
        "goal_locations": goal_locations,
        "description": (
            f"Plan for a grid of size {rows}x{cols} with {num_boxes} boxes. "
            "Each cell has one robot arm. The task is to move each box from its initial "
            "cell to its goal cell through a sequence of legal moves. At every time step, a box can either remain in place, "
            "move to an adjacent cell, or if it is at its goal cell, be marked as 'goal' and remain there."
        )
    }
    return sample

def generate_question_samples(num_samples=5, grid_sizes=None):
    """
    Generate and save a list of question samples.
    """
    if grid_sizes is None:
        grid_sizes = [(4,8, 4, 6), (4,6, 4, 6), (3,6, 4, 6), (2,8, 4, 6), (5,5, 4, 8), (2,8, 3, 5), (2,6, 3, 5)]

    samples = []
    for rows, cols, num_box_low, num_box_high in grid_sizes:
        samples = []
        for i in range(num_samples):
            num_boxes = random.randint(num_box_low, num_box_high)
            sample_id = f"sample_{i+1}"
            sample = synthesize_sample(sample_id, rows, cols, num_boxes)
            samples.append(sample)

            # Save samples locally
            with open(f"/home/ycchen/RoboSteer/dataset_gather/BoxNet1_v2_dataset/question_samples_{rows}_{cols}_{i}.json", "w") as f:
                json.dump(samples, f, indent=2)

    print(f"{len(samples)*7} samples have been saved to 'question_samples.json'.")
    return samples

def read_samples(filename="question_samples.json"):
    """Read and return samples from the saved file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    with open(filename, "r") as f:
        samples = json.load(f)
    return samples

########################################
# 2. Prompt Generation for LLM Testing
########################################

def generate_prompt(sample):
    prompt = f"""You are given the following planning problem:

Grid dimensions: {sample['grid']['rows']} x {sample['grid']['cols']}
Cells: {sample['grid']['cells']}
Adjacency: {json.dumps(sample['grid']['adjacency'], indent=2)}

There is one robot arm in each cell: {sample['arms']}

Boxes: {sample['boxes']}
Initial state: {json.dumps(sample['initial_state'], indent=2)}
Goal locations: {json.dumps(sample['goal_locations'], indent=2)}

Task: Generate a plan as a JSON-formatted list representing the states at successive time steps.
Each state is a dictionary mapping each box to its current cell location.
The plan must satisfy the following:
  1. The first state equals the initial state.
  2. The final state equals the goal state (i.e. each box is located in the same cell as its goal).
  3. Between successive states, a box may either remain in its current cell or move to an adjacent cell (as defined in the adjacency list).
  4. Each cell contains only one arm. Hence, in each cell at most one box can be moved at a time to the adjacent cell.
  5. If a box is at its goal cell, no further action needed for this box. Just keeping it at the goal cell.
  6. Represent each cell state as its current cell location.

In the end of your answer return a list of states and surround it with <<<>>>, such as
<<<[{{"box1": "C1,2", "box2": "C2,3"}}, {{"box1": "C1,3", "box2": "C2,3"}}, ...]>>>.

Your answer:
"""
    return prompt

########################################
# 3. LLM Response Extraction and Correctness Check
########################################

def extract_json_from_response(response):
    """
    Extract the JSON output that is surrounded by markers <<< and >>>.
    Returns the JSON string (without the markers).
    """
    pattern = r"<<<(.*?)>>>"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        raise ValueError("Could not find markers <<< and >>> in the response.")
    json_str = matches[0].strip()
    return json_str

def extract_equation_with_GPT4(response):
    prompt = ('Your task is to extract the final answer from the given answer by another LLM:\n'
              'The final answer should be in the format <<<answer>>>, like <<<[{{"box1": "C1,2", "box2": "C2,3"}}, {{"box1": "C1,3", "box2": "C2,3"}}, ...]>>>.\n'
              'Return only the answer in that format.\n'
              'Input text: ')
    extract_equation = GPT_response('', prompt + response, model_name='gpt-4o', code_interpreter=False,
                                    user_prompt_list=[prompt + response], response_total_list=[], logprobs=False)
    return extract_equation


def check_plan_legality(plan, sample):
    """
    Check that for each pair of consecutive states, every box moves legally.
    A legal move is defined as:
      - If a box does not change its position, that's allowed.
      - If a box moves from one cell to another, the new cell must be adjacent to the previous cell.
      - In a single transition, at most one box may move from the same source cell.
    Returns (True, message) if all transitions are legal; otherwise (False, error message).
    """
    adjacency = sample["grid"]["adjacency"]
    goal_locations = sample["goal_locations"]
    boxes = sample["boxes"]

    for step_idx in range(len(plan) - 1):
        state_curr = plan[step_idx]
        state_next = plan[step_idx + 1]

        # Dictionary to track how many boxes are moved from each cell in this step.
        moved_from_counts = {}

        for box in boxes:
            pos_curr = state_curr.get(box)
            pos_next = state_next.get(box)

            # Once a box is marked "goal", it should remain there.
            if pos_curr == "goal":
                if pos_next != "goal":
                    return False, f"Box '{box}' was marked 'goal' in step {step_idx} but changed in step {step_idx + 1}."

            # Check if the box moved (including moves to "goal")
            if pos_curr != pos_next:
                # Increment the counter for the source cell.
                moved_from_counts[pos_curr] = moved_from_counts.get(pos_curr, 0) + 1

                # Check specific move conditions.
                if pos_next == "goal":
                    if pos_curr != goal_locations[box]:
                        return False, (f"Box '{box}' moved to 'goal' in step {step_idx + 1} but was in {pos_curr}, "
                                       f"not in its goal cell {goal_locations[box]}.")
                else:
                    if pos_curr not in adjacency:
                        return False, f"Invalid current cell {pos_curr} for box '{box}' in step {step_idx}."
                    if pos_next not in adjacency.get(pos_curr, []):
                        return False, (f"Box '{box}' moved from {pos_curr} to {pos_next} in step {step_idx + 1}, "
                                       "which are not adjacent.")

        # Verify that at most one box moved from each cell in this transition.
        for cell, count in moved_from_counts.items():
            if count > 1:
                return False, f"More than one box moved from cell {cell} in step {step_idx + 1}."

    return True, "All moves are legal."


def check_llm_response(response, sample):
    """
    Check the correctness of the LLM response.

    Steps:
      1. Extract the JSON output from between <<< and >>>.
      2. Parse the JSON into a plan (a list of states).
      3. Verify that:
         - The plan is a list with at least 2 steps.
         - Each step is a dictionary containing all boxes.
         - The first step equals the initial state.
         - The final step equals the goal state (each box is at its goal cell or marked as 'goal').
         - Every transition between consecutive states is legal.
    Returns a tuple (is_correct, message).
    """
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
            except Exception as e3:
                return False, f"Could not extract a JSON-formatted plan from the response: {e3}"

    print(f'plan: {plan}')
    print(f'Length of plan: {len(plan)}')

    if not isinstance(plan, list):
        return False, "The extracted JSON is not a list of states."
    if len(plan) < 2:
        return False, f"The plan should have at least 2 steps, but got {len(plan)} steps."

    required_boxes = set(sample["boxes"])
    for i, state in enumerate(plan):
        if not isinstance(state, dict):
            return False, f"Step {i} is not a dictionary."
        if set(state.keys()) != required_boxes:
            return False, (f"Step {i} does not contain the correct boxes. Expected: {required_boxes}, got: {set(state.keys())}")

    if plan[0] != sample["initial_state"]:
        return False, "The first step does not match the initial state."

    if plan[-1] != sample["goal_locations"]:
        return False, f"The final state does not match the goal state. Expected: {sample['goal_locations']}, got: {plan[-1]}"

    legal, message = check_plan_legality(plan, sample)
    if not legal:
        return False, message

    return True, "The LLM response is correctly formatted and all moves are legal."

########################################
# 4. Main: Sample Generation, Prompt, and Test Check
########################################

def run_boxnet1_v2(dataset_input_dir, save_input_dir, gather_save_input_dir, model_name, max_tree_depth, args_path, CodeSteer_LLM):
    print('\n' + '*' * 30)
    print(f'BoxNet1, Model_name: {model_name}, CodeSteer\n')
    base_save_code_dir = save_input_dir + f'/result_boxnet1_{CodeSteer_LLM}_{model_name}_MTD_{max_tree_depth}_CodeSteer_1'

    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    lifted_ratio_list = []
    total_sample_num = 0
    total_correct_num = 0
    grid_sizes = [(4,8, 4, 6), (4,6, 4, 6), (3,6, 4, 6), (2,8, 4, 6), (5,5, 4, 8), (2,8, 3, 5), (2,6, 3, 5)]

    #grid_sizes = [(5, 5, 4, 8), (2, 8, 3, 5), (2, 6, 3, 5)]

    for iteration_num in range(10):
        for rows, cols, num_box_low, num_box_high in grid_sizes:
            total_sample_num += 1

            print('-------###-------###-------###-------')
            print(
                f'Row num is: {rows}, Column num is: {cols}, Iteration num is: {iteration_num}\n\n')

            save_code_dir = os.path.join(base_save_code_dir, f"question_samples_{rows}_{cols}_{iteration_num}/")
            if not os.path.exists(save_code_dir):
                os.makedirs(save_code_dir)

            samples = read_samples(dataset_input_dir + f'/question_samples_{rows}_{cols}_{iteration_num}.json')
            sample = samples[0]
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
                    response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'

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

            is_correct_1, message_1 = check_llm_response(extracted_text_1, sample)
            is_correct_2, message_2 = check_llm_response(extracted_text_2, sample)

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

    run_info = f"CodeSteer, BoxNet1_v2, {CodeSteer_LLM}, {model_name}, MTD_{max_tree_depth}_CodeSteer_1\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}.txt")
    log_run_info(log_file_result, run_info + run_info_result)

# Generate samples if not already created
#_ = generate_question_samples(num_samples=20)

'''
samples = read_samples("question_samples.json")

# For demonstration, choose one sample
for index in range(len(samples)):
    sample = samples[index]
    prompt = generate_prompt(sample)
    print("=== Generated Prompt ===")
    #print(prompt)

    model_name = "gpt-4o"  # Example model name; adjust as needed.
    if model_name in ['o1', "o1-preview", 'o1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo',
                      "claude-3-sonnet-20240229",
                      "claude-3-opus-20240229", "claude-3-haiku-20240307"]:
        response = GPT_response('', prompt, model_name=model_name,
                                code_interpreter=False, user_prompt_list=[prompt],
                                response_total_list=[], logprobs=False)
        print("\n=== LLM Response ===")
        print(response)

    # Check the correctness of the response.
    is_correct, message = check_llm_response(response, sample)
    print("\n=== LLM Response Check ===")
    print("Correct:", is_correct)
    print("Message:", message)
'''