import argparse
import copy
import json
import os
import random
import re
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current working directory to the Python path
sys.path.append(os.getcwd())
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import google.generativeai as genai
from tqdm import tqdm

from combined_prompt import (
    get_init_archive, 
    get_prompt, 
    get_reflexion_prompt
)
from utils import (
    bootstrap_confidence_interval,
    check_math_answer,
    format_question,
    random_id,
    calculate_mmlu_score,
    calculate_math_score,
    check_humaneval_solution
)

import threading
import time
from datetime import datetime

# ----- Cost Tracking -----
class CostTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.session_name = "session"
        self.output_dir = "./results"
        # model -> (input_per_mtok_usd, output_per_mtok_usd)
        # Defaults to env-configurable; if absent, costs compute to 0 while tokens are tracked accurately
        self.model_prices = {}
        self._totals = {}
        self._log_path = None
        self._summary_path = None

    def configure(self, output_dir: str, session_name: str, model: str):
        self.output_dir = output_dir or "./results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.session_name = session_name or "session"
        self._log_path = os.path.join(self.output_dir, f"{self.session_name}_cost_log.jsonl")
        self._summary_path = os.path.join(self.output_dir, f"{self.session_name}_cost_summary.json")
        # load price from env for current model
        # Expect per 1M tokens USD; compute per token in call
        input_per_mtok = os.getenv("GEMINI_FLASH_INPUT_PER_MTOK_USD")
        output_per_mtok = os.getenv("GEMINI_FLASH_OUTPUT_PER_MTOK_USD")
        try:
            if input_per_mtok is not None and output_per_mtok is not None:
                self.model_prices[model] = (float(input_per_mtok), float(output_per_mtok))
        except Exception:
            pass

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int):
        input_usd = output_usd = 0.0
        prices = self.model_prices.get(model)
        if prices:
            in_per_m = prices[0]
            out_per_m = prices[1]
            input_usd = (input_tokens / 1_000_000.0) * in_per_m
            output_usd = (output_tokens / 1_000_000.0) * out_per_m
        return input_usd, output_usd, input_usd + output_usd

    def add_call(self, *, model: str, input_tokens: int, output_tokens: int, label: str, meta: dict | None = None):
        ts = datetime.utcnow().isoformat() + "Z"
        in_cost, out_cost, total_cost = self._estimate_cost(model, input_tokens or 0, output_tokens or 0)
        rec = {
            "timestamp": ts,
            "session": self.session_name,
            "model": model,
            "label": label,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int((input_tokens or 0) + (output_tokens or 0)),
            "input_usd": round(in_cost, 6),
            "output_usd": round(out_cost, 6),
            "total_usd": round(total_cost, 6),
            "meta": meta or {},
        }
        with self.lock:
            # append log
            if self._log_path:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            # update totals
            key = model
            if key not in self._totals:
                self._totals[key] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "input_usd": 0.0, "output_usd": 0.0, "total_usd": 0.0}
            agg = self._totals[key]
            agg["input_tokens"] += rec["input_tokens"]
            agg["output_tokens"] += rec["output_tokens"]
            agg["total_tokens"] += rec["total_tokens"]
            agg["input_usd"] += rec["input_usd"]
            agg["output_usd"] += rec["output_usd"]
            agg["total_usd"] += rec["total_usd"]
            # write summary
            if self._summary_path:
                summary = self.get_summary_unlocked()
                with open(self._summary_path, "w") as f:
                    json.dump(summary, f, indent=2)

    def get_summary_unlocked(self):
        # internal helper assumes lock is held
        return {
            "session": self.session_name,
            "models": self._totals,
            "grand_totals": {
                "input_tokens": sum(v["input_tokens"] for v in self._totals.values()),
                "output_tokens": sum(v["output_tokens"] for v in self._totals.values()),
                "total_tokens": sum(v["total_tokens"] for v in self._totals.values()),
                "input_usd": round(sum(v["input_usd"] for v in self._totals.values()), 6),
                "output_usd": round(sum(v["output_usd"] for v in self._totals.values()), 6),
                "total_usd": round(sum(v["total_usd"] for v in self._totals.values()), 6),
            }
        }

    def get_summary(self):
        with self.lock:
            return self.get_summary_unlocked()

# global tracker
cost_tracker = CostTracker()

client = None

Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])
FORMAT_INST = lambda r: f"Reply with JSON: {str(r)}. Ensure valid JSON."
ROLE_DESC = lambda r: f"You are a {r}."
PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, Exception)
def get_json_response_from_gemini(msg, model, system_message, temperature=0.5):
    response = client.generate_content(
        f"{system_message}\n\n{msg}",
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=4096,
        )
    )
    # token usage (Gemini returns usage metadata when available)
    usage = getattr(response, "usage_metadata", None)
    in_toks = getattr(usage, "prompt_token_count", 0) if usage else 0
    out_toks = getattr(usage, "candidates_token_count", 0) if usage else 0
    cost_tracker.add_call(model=model, input_tokens=in_toks, output_tokens=out_toks, label="get_json_response_from_gemini")

    content = response.text
    
    # Handle empty or None responses
    if not content or content.strip() == "":
        return {
            "thinking": "No response generated", 
            "answer": "No response generated",
            "reasoning": "No response generated",
            "final_answer": "No response generated",
            "notes": "No response generated",
            "refined": "No response generated",
            "statement": "No response generated",
            "voted": "No response generated",
            "final": "No response generated",
            "tests": "No response generated",
            "reason": "No response generated",
            "default_answer": "No response generated"
        }
    
    # Extract JSON from the response
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            return json.loads(json_str)
        else:
            return json.loads(content)
    except json.JSONDecodeError:
        return {
            "thinking": content, 
            "answer": content,
            "reasoning": content,
            "final_answer": content,
            "notes": content,
            "refined": content,
            "statement": content,
            "voted": content,
            "final": content,
            "tests": content,
            "reason": content,
            "default_answer": content
        }


@backoff.on_exception(backoff.expo, Exception)
def preprocess_response(content):
    """Clean and preprocess LLM response before JSON parsing."""
    # Remove markdown code blocks
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*$', '', content)
    
    # Remove leading/trailing whitespace
    content = content.strip()
    
    # If response starts with text, try to find JSON
    if not content.startswith('{'):
        # Look for first { and last }
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            content = content[start:end]
    
    return content



def validate_response(response_dict):
    """Validate response has all required fields and correct types."""
    if not isinstance(response_dict, dict):
        return False
    required_fields = ['thought', 'name', 'code']
    return all(field in response_dict for field in required_fields)



def get_json_response_from_gemini_reflect(msg_list, model, temperature=0.8):
    # Convert OpenAI format to Gemini format
    content = ""
    for msg in msg_list:
        if msg["role"] == "system":
            content += f"System: {msg['content']}\n\n"
        elif msg["role"] == "user":
            content += f"User: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            content += f"Assistant: {msg['content']}\n\n"
    
    response = client.generate_content(
        content,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=4096,
            response_mime_type="application/json"
        )
    )
    usage = getattr(response, "usage_metadata", None)
    in_toks = getattr(usage, "prompt_token_count", 0) if usage else 0
    out_toks = getattr(usage, "candidates_token_count", 0) if usage else 0
    cost_tracker.add_call(model=model, input_tokens=in_toks, output_tokens=out_toks, label="get_json_response_from_gemini_reflect")
    
    content = preprocess_response(response.text)
    json_candidates = []
    start = content.find('{')
    end = content.rfind('}') + 1
    if start != -1 and end != 0:
        json_candidates.append(content[start:end])
    if "```json" in content:
        parts = content.split("```json")
        if len(parts) > 1:
            json_part = parts[1].split("```")[0]
            json_candidates.append(json_part.strip())
    json_candidates.append(content)
    for candidate in json_candidates:
        try:
            json_dict = json.loads(candidate)
            if all(field in json_dict for field in ['thought', 'name', 'code']):
                if not isinstance(json_dict['code'], str):
                    raise ValueError("'code' field must be a string")
                return json_dict
        except:
            continue
    return {"error": "Failed to parse JSON", "raw_content": content}


class LLMAgentBase:
    def __init__(
        self,
        output_fields: list,
        agent_name: str,
        role="helpful assistant",
        model="gemini-2.0-flash",
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
            js = get_json_response_from_gemini(prompt, self.model, sys, self.temperature)
        except Exception as e:
            if "context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("Context too long.")
            js = {k: "" for k in self.output_fields}
        
        # Ensure all required output fields are present
        result = []
        for field in self.output_fields:
            if field in js:
                result.append(Info(field, repr(self), js[field], iteration_idx))
            else:
                # Provide fallback for missing fields
                result.append(Info(field, repr(self), "", iteration_idx))
        
        return result

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx)


class AgentSystem:
    def forward(self, taskInfo):
        raise NotImplementedError


def search(args):
    archive_path = os.path.join(args.save_dir, f"{args.expr_name}_archive.json")
    archive = get_init_archive()
    start_gen = 0
    if os.path.exists(archive_path):
        with open(archive_path, "r") as f:
            archive = json.load(f)
        if archive and "generation" in archive[-1]:
            start_gen = archive[-1]["generation"]
            if isinstance(start_gen, str) and start_gen == "initial":
                start_gen = 0
            else:
                start_gen = int(start_gen)

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
            next_sol = get_json_response_from_gemini_reflect(msg_list, args.model)
            
            # ROBUST REFLEXION WITH VALIDATION AT EACH STEP
            p1, p2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            
            # First reflexion step
            msg_list.extend([
                {"role": "assistant", "content": str(next_sol)},
                {"role": "user", "content": p1},
            ])
            
            reflexion_sol = get_json_response_from_gemini_reflect(msg_list, args.model)
            next_sol = reflexion_sol
            
            # Second reflexion step
            msg_list.extend([
                {"role": "assistant", "content": str(next_sol)},
                {"role": "user", "content": p2},
            ])
            
            final_sol = get_json_response_from_gemini_reflect(msg_list, args.model)
            next_sol = final_sol
                
        except Exception as e:
            print(f"LLM generation failed: {e}")
            n -= 1
            continue

        acc = []
        for debug_attempt in range(args.debug_max):
            try:
                acc = evaluate_forward_fn(args, next_sol["code"])
                if np.mean(acc) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                msg_list.append({"role": "assistant", "content": str(next_sol)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_sol = get_json_response_from_gemini_reflect(msg_list, args.model)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        if not acc:
            n -= 1
            continue

        fitness_str = bootstrap_confidence_interval(acc)
        next_sol['fitness'] = fitness_str
        next_sol['generation'] = n + 1

        if 'debug_thought' in next_sol:
            del next_sol['debug_thought']
        if 'reflection' in next_sol:
            del next_sol['reflection']
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
    # Add necessary classes to the global namespace so they're available when operators are executed
    globals_dict = globals().copy()
    globals_dict.update({
        'LLMAgentBase': LLMAgentBase,
        'Info': Info
    })
    
    # Add syntax validation before execution
    try:
        compile(forward_str, '<string>', 'exec')
    except SyntaxError as e:
        print(f"Syntax error in generated code: {e}")
        raise Exception(f"Syntax error: {e}")
    
    # Validate imports - only allow safe, built-in modules
    allowed_imports = {
        'json', 're', 'random', 'math', 'collections', 'typing', 
        'numpy', 'np', 'string', 'itertools', 'functools', 'datetime',
        'copy', 'sys', 'os', 'traceback'
    }
    
    import_lines = [line.strip() for line in forward_str.split('\n') if line.strip().startswith(('import ', 'from '))]
    for import_line in import_lines:
        # Extract module name from import statement
        if import_line.startswith('import '):
            module = import_line[7:].split()[0].split('.')[0]
        elif import_line.startswith('from '):
            module = import_line[5:].split()[0].split('.')[0]
        else:
            continue
            
        if module not in allowed_imports:
            print(f"Disallowed import detected: {import_line}")
            print(f"Module '{module}' is not in the allowed list: {allowed_imports}")
            raise Exception(f"Disallowed import: {module}")
    
    try:
        exec(forward_str, globals_dict, namespace)
        names = list(namespace.keys())
        if len(names) != 1:
            raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
        func = namespace[names[0]]
        if not callable(func):
            raise AssertionError(f"{func} is not callable")
        setattr(AgentSystem, "forward", func)
    except Exception as e:
        print(f"Error setting up forward function: {e}")
        print(f"Generated code:\n{forward_str}")
        raise Exception(f"Code execution error: {e}")

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
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    for i, res in enumerate(results):
        try:
            pred = res.content if isinstance(res, Info) else res
            ex = examples[i]
            correct = 0
            if ex["type"] == "mmlu":
                correct = calculate_mmlu_score(ex["answer"], pred)
            elif ex["type"] == "math":
                correct = calculate_math_score(ex["solution"], pred)
            elif ex["type"] == "humaneval":
                # Use the advanced HumanEval parsing with execution testing
                if isinstance(pred, str) and "def " in pred:
                    correct = check_humaneval_solution(
                        pred,
                        ex.get("test", ""),
                        ex.get("entry_point", "solution"),
                        ex.get("canonical_solution", ""),
                    )
            acc_list.append(correct)
            
        except Exception as e:
            acc_list.append(0)
            continue

    # Calculate breakdown by question type
    type_results = {"mmlu": [], "math": [], "humaneval": []}
    for i, res in enumerate(results):
        try:
            pred = res.content if isinstance(res, Info) else res
            ex = examples[i]
            correct = 0
            if ex["type"] == "mmlu":
                correct = calculate_mmlu_score(ex["answer"], pred)
            elif ex["type"] == "math":
                correct = calculate_math_score(ex["solution"], pred)
            elif ex["type"] == "humaneval":
                if isinstance(pred, str) and "def " in pred:
                    correct = check_humaneval_solution(pred, ex.get("test", ""), ex.get("entry_point", "solution"))
            
            type_results[ex["type"]].append(correct)
        except Exception as e:
            type_results[ex["type"]].append(0)
            continue

    # Calculate and display breakdown
    breakdown_str = ""
    for qtype, results_list in type_results.items():
        if results_list:
            type_accuracy = sum(results_list) / len(results_list) * 100
            type_successful = sum(results_list)
            type_total = len(results_list)
            breakdown_str += f"{qtype}: {type_successful}/{type_total} ({type_accuracy:.1f}%)"
        else:
            breakdown_str += f"{qtype}: 0/0 (0.0%)"
        breakdown_str += ", "

    successful_questions = len([x for x in acc_list if x > 0])
    total_questions = len(acc_list)
    print(f"Evaluation completed: {successful_questions}/{total_questions} questions successful")
    print(f"Breakdown: {breakdown_str.rstrip(', ')}")
    print(f"Accuracy: {bootstrap_confidence_interval(acc_list)}")
    # Print cost summary
    summary = cost_tracker.get_summary()
    gt = summary.get("grand_totals", {})
    print(f"Cost so far — tokens: in {gt.get('input_tokens',0)}, out {gt.get('output_tokens',0)}, total {gt.get('total_tokens',0)}; USD: in ${gt.get('input_usd',0):.4f}, out ${gt.get('output_usd',0):.4f}, total ${gt.get('total_usd',0):.4f}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../dataset/combined_train.jsonl")
    parser.add_argument("--test_data", type=str, default="../dataset/combined_test.jsonl")
    parser.add_argument("--valid_size", type=int, default=100)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="./results/")
    parser.add_argument("--expr_name", type=str, default="combined_gemini")
    parser.add_argument("--n_generation", type=int, default=20)
    parser.add_argument("--debug_max", type=int, default=2)
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--api_key", type=str, default=None)

    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via --api_key argument or GOOGLE_API_KEY environment variable")
    
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(args.model)

    # configure cost tracker
    cost_tracker.configure(output_dir=args.save_dir, session_name=args.expr_name, model=args.model)

    # SEARCHING_MODE = True
    # search(args)

    SEARCHING_MODE = False
    evaluate(args)
