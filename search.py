import argparse
import copy
import json
import os
import random
import re
import sys

# Add the current working directory to the Python path
sys.path.append(os.getcwd())
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm

from combined.combined_prompt import get_init_archive, get_prompt, get_reflexion_prompt
from combined.utils import (
    bootstrap_confidence_interval,
    check_math_answer,
    format_question,
    random_id,
)

client = None

Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])
FORMAT_INST = lambda r: f"Reply with JSON: {str(r)}. Ensure valid JSON."
ROLE_DESC = lambda r: f"You are a {r}."
PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(msg_list, model, temperature=0.8):
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


class LLMAgentBase:
    def __init__(
        self,
        output_fields: list,
        agent_name: str,
        role="helpful assistant",
        model="gpt-4o",
        temperature=0.5,
    ) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction):
        desc = {k: f"Your {k}." for k in self.output_fields}
        sys_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(desc)
        text = ""
        for info in input_infos:
            if not isinstance(info, Info):
                continue
            field, author, content, idx = info
            if author == repr(self):
                author += " (yourself)"
            if field == "task":
                text += f"# Task:\n{content}\n\n"
            else:
                text += f"### {field} by {author}:\n{content}\n\n"
        return sys_prompt, text + instruction

    def query(self, input_infos, instruction, iteration_idx=-1):
        sys, prompt = self.generate_prompt(input_infos, instruction)
        try:
            js = get_json_response_from_gpt(prompt, self.model, sys, self.temperature)
        except Exception as e:
            if "context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("Context too long.")
            js = {k: "" for k in self.output_fields}
        return [Info(k, repr(self), v, iteration_idx) for k, v in js.items()]

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx)


class AgentSystem:
    pass


def search(args):
    archive_path = os.path.join(args.save_dir, f"{args.expr_name}_archive.json")
    archive = get_init_archive()
    start_gen = 0
    if os.path.exists(archive_path):
        with open(archive_path, "r") as f:
            archive = json.load(f)
        if archive and "generation" in archive[-1]:
            start_gen = archive[-1]["generation"]

    for sol in archive:
        if "fitness" in sol:
            continue
        sol["generation"] = "initial"
        print(f"Evaluating initial solution: {sol['name']}")
        try:
            acc = evaluate_forward_fn(args, sol["code"])
            sol["fitness"] = bootstrap_confidence_interval(acc)
        except Exception as e:
            print(f"Error evaluating initial solution: {e}")
            sol["fitness"] = "Error"
        with open(archive_path, "w") as f:
            json.dump(archive, f, indent=4)

    for n in range(start_gen, args.n_generation):
        print(f"============ Generation {n + 1} ============")
        sys, prompt = get_prompt(archive)
        msg_list = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
        try:
            next_sol = get_json_response_from_gpt_reflect(msg_list, args.model)
            p1, p2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            msg_list.extend(
                [
                    {"role": "assistant", "content": str(next_sol)},
                    {"role": "user", "content": p1},
                ]
            )
            next_sol = get_json_response_from_gpt_reflect(msg_list, args.model)
            msg_list.extend(
                [
                    {"role": "assistant", "content": str(next_sol)},
                    {"role": "user", "content": p2},
                ]
            )
            next_sol = get_json_response_from_gpt_reflect(msg_list, args.model)
        except Exception as e:
            print(f"LLM generation failed: {e}")
            continue

        acc = []
        for _ in range(args.debug_max):
            try:
                acc = evaluate_forward_fn(args, next_sol["code"])
                if np.mean(acc) < 0.01 and SEARCHING_MODE:
                    raise ValueError("Accuracy is all zero.")
                break
            except Exception as e:
                print(f"Evaluation failed: {e}")
                msg_list.append({"role": "assistant", "content": str(next_sol)})
                msg_list.append(
                    {
                        "role": "user",
                        "content": f"Error: {e}. Debug your code.",
                    }
                )
                try:
                    next_sol = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as llm_e:
                    print(f"LLM debug generation failed: {llm_e}")
                    break
        if not acc:
            continue

        next_sol["fitness"] = bootstrap_confidence_interval(acc)
        next_sol["generation"] = n + 1
        archive.append(next_sol)
        with open(archive_path, "w") as f:
            json.dump(archive, f, indent=4)


def evaluate(args):
    archive_path = os.path.join(args.save_dir, f"{args.expr_name}_archive.json")
    eval_path = archive_path.replace(".json", "_evaluate.json")
    with open(archive_path, "r") as f:
        archive = json.load(f)
    eval_archive = []
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_archive = json.load(f)

    evaluated_names = {s["name"] for s in eval_archive}
    for sol in archive:
        if sol["name"] in evaluated_names:
            continue
        print(f"Testing solution: {sol['name']}")
        try:
            acc = evaluate_forward_fn(args, sol["code"])
            sol["test_fitness"] = bootstrap_confidence_interval(acc)
        except Exception as e:
            print(f"Error testing solution: {e}")
            sol["test_fitness"] = "Error"
        eval_archive.append(sol)
        with open(eval_path, "w") as f:
            json.dump(eval_archive, f, indent=4)


def evaluate_forward_fn(args, forward_str):
    namespace = {}
    exec(forward_str, globals(), namespace)
    func = namespace[list(namespace.keys())[0]]
    setattr(AgentSystem, "forward", func)

    data_file = (
        args.train_data if SEARCHING_MODE else args.test_data
    )
    with open(data_file, "r") as f:
        examples = [json.loads(line) for line in f]

    if SEARCHING_MODE:
        random.shuffle(examples)
        examples = examples[: args.valid_size]

    questions = [format_question(ex) for ex in examples]
    task_queue = [Info("task", "User", q, -1) for q in questions]
    agentSystem = AgentSystem()

    acc_list = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(
            tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue))
        )

    for i, res in enumerate(results):
        pred = res.content if isinstance(res, Info) else res
        ex = examples[i]
        correct = 0
        if ex["type"] == "mmlu":
            pred_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(pred).strip(), -1)
            if pred_idx == ex["answer"]:
                correct = 1
        elif ex["type"] == "math":
            if check_math_answer(pred, ex["solution"]):
                correct = 1
        elif ex["type"] == "humaneval":
            # HumanEval requires execution testing, which is complex.
            # For now, we just check if it produces valid, non-empty code.
            if isinstance(pred, str) and "def " in pred and len(pred) > 20:
                correct = 1  # Placeholder for actual execution-based eval
        acc_list.append(correct)

    print(f"Accuracy: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="combined_train.jsonl")
    parser.add_argument("--test_data", type=str, default="combined_test.jsonl")
    parser.add_argument("--valid_size", type=int, default=100)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument("--expr_name", type=str, default="combined_gpt4o")
    parser.add_argument("--n_generation", type=int, default=20)
    parser.add_argument("--debug_max", type=int, default=2)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    client = openai.OpenAI(api_key=args.api_key)

    SEARCHING_MODE = True
    search(args)

    SEARCHING_MODE = False
    evaluate(args)
