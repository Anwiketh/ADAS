import random
import string
import numpy as np
import re

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

def check_math_answer(predicted, ground_truth):
    # Basic normalization for math answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    return pred_norm == gt_norm
