import json

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nYour reasoning and the overall concept behind the agent design.\n**Implementation:**\nDescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, taskInfo):
    # Your code here. You will receive a `taskInfo` that contains one of:
    # - A multiple-choice question (MMLU-style) with options (A)-(D). Return ONLY 'A' or 'B' or 'C' or 'D'.
    # - A math problem. Return ONLY the final answer (number/fraction/single word like odd/even/neither) without steps.
    # - A HumanEval coding task prompt for a Python function. Return ONLY valid Python code that defines the requested function.
    return "A"
"""
}

COT = {
    "thought": "By encouraging the LLM to think step by step rather than directly outputting an answer, Chain-of-Thought improves reasoning and provides intermediate steps for challenging tasks.",
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Generic CoT instruction that works across modalities
    cot_instruction = (
        "Think step by step, then provide ONLY the final result in the required format:\\n"
        "- For multiple-choice return exactly one of A/B/C/D.\\n"
        "- For math return only the final numeric/simplified answer.\\n"
        "- For HumanEval return only valid Python code defining the requested function."
    )
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
    thinking, answer = cot_agent([taskInfo], cot_instruction)
    return answer
"""
}

COT_SC = {
    "thought": "Self-Consistency with Chain-of-Thought: sample multiple reasoning paths at higher temperature and aggregate.",
    "name": "Self-Consistency with CoT",
    "code": """def forward(self, taskInfo):
    cot_instruction = (
        "Think step by step, then provide ONLY the final result in the required format:\\n"
        "- For multiple-choice return exactly one of A/B/C/D.\\n"
        "- For math return only the final numeric/simplified answer.\\n"
        "- For HumanEval return only valid Python code defining the requested function."
    )
    N = 5
    from collections import Counter
    agents = [LLMAgentBase(['thinking', 'answer'], 'CoT Agent', temperature=0.8) for _ in range(N)]
    answers = []
    for i in range(N):
        thinking, ans = agents[i]([taskInfo], cot_instruction)
        answers.append(ans.content if hasattr(ans, 'content') else ans)

    # For HumanEval, prefer the longest valid-looking code to avoid empty strings; for others, majority vote
    if any(isinstance(a, str) and ("def " in a) for a in answers):
        answers_code = [a for a in answers if isinstance(a, str) and ("def " in a)]
        answers_code.sort(key=lambda x: len(x), reverse=True)
        return answers_code[0]
    else:
        return Counter(answers).most_common(1)[0][0]
"""
}

Reflexion = {
    "thought": "Self-refine by iteratively improving the answer using feedback.",
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, taskInfo):
    initial_instruction = (
        "Think step by step, then provide ONLY the final result in the required format:\\n"
        "- For multiple-choice return exactly one of A/B/C/D.\\n"
        "- For math return only the final numeric/simplified answer.\\n"
        "- For HumanEval return only valid Python code defining the requested function."
    )
    refine_instruction = (
        "Given the prior attempt and feedback, carefully revise and provide ONLY the final result in the required format."
    )
    critic_instruction = (
        "Critique the above answer precisely. If it is certainly correct, set 'correct' to True; otherwise, provide feedback."
    )

    cot_agent = LLMAgentBase(['thinking', 'answer'], 'CoT Agent')
    critic = LLMAgentBase(['feedback', 'correct'], 'Critic Agent')

    thinking, answer = cot_agent([taskInfo], initial_instruction, 0)
    N_max = 4
    inputs = [taskInfo, thinking, answer]

    for i in range(N_max):
        feedback, correct = critic(inputs, critic_instruction, i)
        if str(correct.content).strip().lower() == 'true':
            return answer
        thinking, answer = cot_agent(inputs, refine_instruction, i + 1)
        inputs.extend([feedback, thinking, answer])
    return answer
"""
}

StepBack = {
    "thought": "Step-back abstraction: elicit principles or meta-thought before solving. Helps on math and MCQ, and can plan for coding.",
    "name": "Step-back Abstraction",
    "code": """def forward(self, taskInfo):
    principle_instruction = (
        "Identify the underlying principles or plan relevant to this task. Reason step by step."
    )
    solve_instruction = (
        "Using the above, provide ONLY the final result in the required format:\\n"
        "- For multiple-choice return exactly one of A/B/C/D.\\n"
        "- For math return only the final numeric/simplified answer.\\n"
        "- For HumanEval return only valid Python code defining the requested function."
    )
    principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')
    solver = LLMAgentBase(['thinking', 'answer'], 'Solver Agent')

    thinking_p, principle = principle_agent([taskInfo], principle_instruction)
    thinking_s, answer = solver([taskInfo, thinking_p, principle], solve_instruction)
    return answer
"""
}

system_prompt = "You are a helpful assistant. Make sure to return a WELL-FORMED JSON object."

base = """# Overview
You are designing agentic systems to perform well on a COMBINED benchmark that mixes:
- MMLU multiple-choice questions (return exactly one of A/B/C/D),
- MATH problems (return only the final answer, numeric or simplified fraction/word),
- HumanEval code generation tasks (return only valid Python code defining the requested function).

Your 'forward(self, taskInfo)' should robustly solve any of the above, based only on the provided task text.

# The utility code
```python
import json
import backoff
import openai
from collections import namedtuple
from utils import random_id

client = openai.OpenAI()
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request: f"Reply EXACTLY with the following JSON format.\\n{str(request)}\\nEnsure WELL-FORMED JSON and DO NOT MISS ANY FIELDS."
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system_message},{"role":"user","content":msg}],
        temperature=temperature,
        max_tokens=2048,
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

class LLMAgentBase:
    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='gpt-4o-2024-05-13', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction):
        fmt_desc = {k: f"Your {k}." for k in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\\n\\n" + FORMAT_INST(fmt_desc)

        text = ''
        for info in input_infos:
            if isinstance(info, Info):
                field, author, content, idx = info
            else:
                continue
            if author == self.__repr__():
                author += " (yourself)"
            if field == 'task':
                text += f"# Your Task:\\n{content}\\n\\n"
            elif idx != -1:
                text += f"### {field} #{idx+1} by {author}:\\n{content}\\n\\n"
            else:
                text += f"### {field} by {author}:\\n{content}\\n\\n"

        return system_prompt, text + instruction

    def query(self, input_infos, instruction, iteration_idx=-1):
        sys, prompt = self.generate_prompt(input_infos, instruction)
        js = get_json_response_from_gpt(prompt, self.model, sys, self.temperature)
        outs = []
        for k, v in js.items():
            outs.append(Info(k, self.__repr__(), v, iteration_idx))
        return outs

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx)
```

# Discovered architecture archive
[ARCHIVE]

Fitness is median and 95% Bootstrap CI of accuracy on validation. Maximize fitness.

# Output Instruction
- "thought": Your reasoning and design for the next agent.
- "name": Name of your agent.
- "code": EXACT Python code for the single method `forward(self, taskInfo)` implementing your design. Always return ONLY the final result in the correct format for the task type.
"""

Reflexion_prompt_1 = f""""[EXAMPLE]Carefully review the proposed agent and reflect on:
1) Interestingness vs. archive; 2) Implementation mistakes; 3) Concrete improvements without changing overall idea unless necessary.
Then revise and provide updated code."""
Reflexion_prompt_2 = """Using the 'WRONG Implementation examples' tips, revise the code further. Return 'reflection', then repeat 'thought' and 'name', and updated 'code'."""

def get_init_archive():
    # A compact but diverse starter set
    return [COT, COT_SC, Reflexion, StepBack]

def get_prompt(current_archive, adaptive=False):
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    return system_prompt, prompt

def get_reflexion_prompt(prev_example):
    prev_str = "Here is the previous agent you tried:\\n" + json.dumps(prev_example) + "\\n\\n" if prev_example else ""
    r1 = Reflexion_prompt_1.replace("[EXAMPLE]", prev_str)
    return r1, Reflexion_prompt_2
