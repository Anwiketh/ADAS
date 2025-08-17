import random
import string
import numpy as np
import re
import regex
import math
import json
import threading
import time
from typing import Any, Dict, List, Optional, Union
import sympy
from sympy import N, simplify

def random_id(length=4):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

def bootstrap_confidence_interval(data, num_samples=100000, level=0.95):
    data = np.array(data)
    means = []
    for _ in range(num_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    
    means = np.array(means)
    lower_p = (1.0 - level) / 2.0
    upper_p = 1.0 - lower_p
    lower = np.percentile(means, lower_p * 100)
    upper = np.percentile(means, upper_p * 100)
    median = np.median(means)
    
    return f"95% Bootstrap CI: ({lower*100:.1f}%, {upper*100:.1f}%), Median: {median*100:.1f}%"

def format_question(item):
    if item['type'] == 'mmlu':
        q = f"Question: {item['question']}\n"
        choices = "\\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(item['choices'])])
        return f"{q}{choices}"
    elif item['type'] == 'math':
        return f"Solve the following math problem:\n{item['problem']}"
    elif item['type'] == 'humaneval':
        return f"Complete the following Python function:\n{item['prompt']}"
    return ""

def normalize_answer(ans):
    if isinstance(ans, str):
        # Remove common formatting and normalize whitespace
        ans = re.sub(r'\\boxed\{(.+?)\}', r'\1', ans)
        ans = ans.strip()
    return ans

def extract_string_from_output(output: Any) -> str:
    """Extract a string from various output formats."""
    if isinstance(output, str):
        return output
    elif isinstance(output, (list, tuple)):
        if output:
            first_item = output[0]
            if isinstance(first_item, str):
                return first_item
            elif hasattr(first_item, 'content'):
                return str(first_item.content)
            else:
                return str(first_item)
        return ""
    elif isinstance(output, dict):
        # Try to find common answer fields
        for key in ['answer', 'result', 'solution', 'final_answer', 'output']:
            if key in output:
                return str(output[key])
        return str(output)
    else:
        return str(output)

def calculate_mmlu_score(expected_output: int, prediction: str) -> int:
    """Calculate MMLU score by extracting answer choice from prediction."""
    prediction_str = extract_string_from_output(prediction)
    
    # Use regex to find "The final answer is: [A-Z]"
    match = regex.search(r"The final answer is: ([A-Z])", prediction_str, regex.IGNORECASE)
    if match:
        answer_letter = match.group(1).upper()
        predicted_idx = ord(answer_letter) - ord('A')
        return 1 if predicted_idx == expected_output else 0
    
    # Fallback: look for standalone A, B, C, D
    match = regex.search(r"\\b([A-D])\\b", prediction_str, regex.IGNORECASE)
    if match:
        answer_letter = match.group(1).upper()
        predicted_idx = ord(answer_letter) - ord('A')
        return 1 if predicted_idx == expected_output else 0
    
    return 0

def extract_model_answer(text: str) -> str:
    """Extract the model's answer from the text."""
    # Try to extract from XML-like tags first
    match = regex.search(r"<final_answer>(.*?)</final_answer>", text, regex.DOTALL | regex.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    match = regex.search(r"<answer>(.*?)</answer>", text, regex.DOTALL | regex.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try to extract from LaTeX boxed
    match = regex.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()
    
    # Try to extract from the last sentence
    sentences = text.split('.')
    if sentences:
        last_sentence = sentences[-1].strip()
        if last_sentence and len(last_sentence) > 0:
            return last_sentence
    
    return text.strip()

def math_equal(prediction: str, reference: str) -> bool:
    """Check if two math expressions are equal."""
    try:
        # Normalize both predicted and reference answers
        pred = extract_model_answer(prediction)
        ref = extract_model_answer(reference)
        
        # Normalize common formatting
        pred = normalize_answer(pred)
        ref = normalize_answer(ref)
        
        # Try numerical comparison first
        try:
            pred_num = float(parse_digits(pred)) if is_digit(parse_digits(pred)) else None
            ref_num = float(parse_digits(ref)) if is_digit(parse_digits(ref)) else None
            if pred_num is not None and ref_num is not None:
                return math.isclose(pred_num, ref_num, rel_tol=1e-6, abs_tol=1e-6)
        except ValueError:
            pass
        
        # Try symbolic comparison
        return symbolic_equal(pred, ref)
    except:
        return False

def is_digit(num: str) -> bool:
    """Check if a string represents a digit."""
    try:
        float(num)
        return True
    except ValueError:
        return False

def parse_digits(num: str) -> str:
    """Parse digits from a string, handling commas and percentages."""
    num = num.replace(',', '').replace('%', '')
    return num.strip()

def symbolic_equal(a: str, b: str) -> bool:
    """Check if two symbolic expressions are equal."""
    try:
        # Try parsing as LaTeX first
        try:
            expr_a = sympy.parse_latex(a)
            expr_b = sympy.parse_latex(b)
        except:
            # Fall back to regular parsing
            expr_a = sympy.parse_expr(a)
            expr_b = sympy.parse_expr(b)
        
        # Simplify and compare
        diff = simplify(expr_a - expr_b)
        return diff == 0
    except:
        return False

def calculate_math_score(expected_output: str, prediction: str) -> int:
    """Calculate math score by comparing prediction with expected output."""
    try:
        return 1 if math_equal(prediction, expected_output) else 0
    except:
        return 0

def run_with_timeout(func, args, timeout=10):
    """Run a function with a timeout."""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise TimeoutError(f"Function timed out after {timeout} seconds")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]

def extract_code_block(text: str) -> str:
    """Extract Python code from a fenced code block if present; otherwise return the text unchanged."""
    try:
        # ```python ... ``` or ``` ... ```
        m = regex.search(r"```(?:python)?\s*([\s\S]*?)```", text, regex.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text
    except Exception:
        return text


def extract_function_code(text: str, entry_point: str) -> str:
    """Extract just the function definition block for the given entry_point if present."""
    try:
        # Ensure we are working with raw code (no fences)
        code = extract_code_block(text)
        # Find def entry_point(...): and capture until next def at same indent or EOF
        pattern = rf"(^\s*def\s+{regex.escape(entry_point)}\s*\(.*\):[\s\S]*?)(?=^\s*def\s+|\Z)"
        m = regex.search(pattern, code, regex.MULTILINE)
        if m:
            return m.group(1).strip()
        return code
    except Exception:
        return text


def check_humaneval_solution(solution: str, test: str, entry_point: str, canonical_solution: str = "") -> int:
    """Check if a HumanEval solution passes the provided test code.
    - Strips code fences from the solution if present
    - Executes the full solution (to preserve any helper code)
    - Loads the entry_point function and passes it into check(candidate) when required
    - Runs the test with a timeout to avoid hangs
    A run that completes without raising is considered PASS (HumanEval tests use asserts and return None).
    If the run fails, prints the candidate code and the dataset canonical solution to aid debugging.
    """
    try:
        import inspect
        from typing import List, Dict, Tuple, Optional, Any

        namespace: Dict[str, Any] = {}

        helper_code = """
import math
import random
import re
import string
import sys
import time
import itertools
from collections import defaultdict, Counter
"""
        processed_solution = extract_code_block(solution)

        namespace.update({
            "List": List,
            "Dict": Dict,
            "Tuple": Tuple,
            "Optional": Optional,
            "Any": Any,
        })

        exec(helper_code, namespace)
        exec(processed_solution, namespace)

        func = namespace.get(entry_point)
        if not callable(func):
            print(f"HumanEval: entry point '{entry_point}' not found or not callable")
            # Show candidate and canonical to help debugging
            try:
                candidate_snippet = extract_function_code(processed_solution, entry_point)
            except Exception:
                candidate_snippet = processed_solution
            print(f"HumanEval candidate code for '{entry_point}':\n{candidate_snippet}")
            if canonical_solution:
                print(f"HumanEval canonical solution for '{entry_point}':\n{canonical_solution}")
            return 0

        exec(test, namespace)
        check_fn = namespace.get('check')
        if not callable(check_fn):
            print("HumanEval: 'check' function not defined in test")
            try:
                candidate_snippet = extract_function_code(processed_solution, entry_point)
            except Exception:
                candidate_snippet = processed_solution
            print(f"HumanEval candidate code for '{entry_point}':\n{candidate_snippet}")
            if canonical_solution:
                print(f"HumanEval canonical solution for '{entry_point}':\n{canonical_solution}")
            return 0

        def _invoke_check():
            sig = inspect.signature(check_fn)
            if len(sig.parameters) == 1:
                check_fn(func)
            else:
                check_fn()
            return True

        try:
            result = run_with_timeout(_invoke_check, args=(), timeout=15)
            return 1 if result else 0
        except Exception as e:
            print(f"HumanEval: tests failed for '{entry_point}': {e}")
            try:
                candidate_snippet = extract_function_code(processed_solution, entry_point)
            except Exception:
                candidate_snippet = processed_solution
            print(f"HumanEval candidate code for '{entry_point}':\n{candidate_snippet}")
            if canonical_solution:
                print(f"HumanEval canonical solution for '{entry_point}':\n{canonical_solution}")
            return 0
    except Exception as e:
        print(f"HumanEval: execution error for '{entry_point}': {e}")
        try:
            processed_solution = extract_code_block(solution)
            candidate_snippet = extract_function_code(processed_solution, entry_point)
        except Exception:
            candidate_snippet = solution
        print(f"HumanEval candidate code for '{entry_point}':\n{candidate_snippet}")
        if canonical_solution:
            print(f"HumanEval canonical solution for '{entry_point}':\n{canonical_solution}")
        return 0

def check_math_answer(predicted, ground_truth):
    """Updated to use the new advanced math parsing."""
    return calculate_math_score(ground_truth, predicted) == 1
