import json

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, taskInfo):
    # Your code here
    return answer
"""
}

CoT = {
    "thought": "Chain-of-Thought reasoning that encourages step-by-step thinking to derive answers.",
    "name": "CoT",
    "code": """def forward(self, taskInfo):
    # Instruction for Chain-of-Thought reasoning
    cot_instruction = "You are a careful reasoner. Think step by step and derive a concise final answer.\\n\\nProblem:\\n{problem}\\n\\nContext:\\n{context}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).\\n - If the problem is mathematics, also append a final line with the canonical math format: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<reasoning><![CDATA[\\nYour step-by-step reasoning here.\\n]]></reasoning>\\n<final_answer><![CDATA[\\nYour final answer here.\\n]]></final_answer>\\n</response>"
    
    cot_agent = LLMAgentBase(['reasoning', 'final_answer'], 'CoT Agent')
    reasoning, final_answer = cot_agent([taskInfo], cot_instruction.format(
        problem=taskInfo.content,
        context=""
    ))
    return final_answer.content if hasattr(final_answer, 'content') else final_answer
"""
}

Debate = {
    "thought": "LLM-Debate with three debaters and up to two rounds to converge on better solutions.",
    "name": "Debate",
    "code": """def forward(self, taskInfo):
    # Round 1: Three debaters
    debaters = []
    debate_instruction = "You are Debater {debater_id} in a 3-debater, 2-round debate. Argue towards the correct answer.\\n\\nProblem:\\n{problem}\\n\\nRound {round_num} Statement:\\n<response>\\n<statement><![CDATA[\\nYour short statement.\\n]]></statement>\\n</response>"
    
    for i in range(3):
        debater = LLMAgentBase(['statement'], f'Debater {i+1}')
        statement = debater([taskInfo], debate_instruction.format(
            debater_id=i+1,
            problem=taskInfo.content,
            round_num=1
        ))
        debaters.append(statement[0])
    
    # Round 2: Three debaters respond to others
    round2_statements = []
    for i in range(3):
        debater = LLMAgentBase(['statement'], f'Debater {i+1}')
        context = f"Previous statements: {[s.content for s in debaters]}"
        statement = debater([taskInfo], debate_instruction.format(
            debater_id=i+1,
            problem=taskInfo.content,
            round_num=2
        ))
        round2_statements.append(statement[0])
    
    # Synthesis
    transcript = f"Round 1: {[s.content for s in debaters]}\\nRound 2: {[s.content for s in round2_statements]}"
    synthesis_instruction = "Given the following debate transcript, synthesize the key arguments presented by each side and identify the core points of contention. Based on this synthesis, provide a concise and definitive answer to the central question debated. The Combined dataset often involves nuanced arguments and requires careful consideration of context to avoid misrepresentation. Pay close attention to identifying the underlying assumptions and logical fallacies employed by each side.\\n\\nTranscript:\\n{transcript}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X.\\n - If the problem is mathematics, also append a final line: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task, output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<final_answer><![CDATA[\\nThe answer is ...\\n]]></final_answer>\\n</response>"
    
    synthesizer = LLMAgentBase(['final_answer'], 'Synthesis Agent')
    final_answer = synthesizer([taskInfo], synthesis_instruction.format(transcript=transcript))
    return final_answer[0].content if hasattr(final_answer[0], 'content') else final_answer[0]
"""
}

SelfConsistency = {
    "thought": "Aggregate five CoT reasoning paths and vote for the most consistent answer.",
    "name": "SelfConsistency",
    "code": """def forward(self, taskInfo):
    # Generate five independent answers
    answers = []
    cot_instruction = "Generate five independent CoT answers.\\n\\nProblem:\\n{problem}\\n\\nContext:\\n{context}\\n\\nReturn five <answer> blocks.\\nEach <answer> must be ONLY the exact answer string (no extra text, no quotes or year).\\nIf the question is multiple-choice (A/B/C/D), the final chosen answer should be provided later in the vote step as: The final answer is: X.\\nIf the problem is mathematics, ensure each answer is suitable to be placed inside $$\\\\boxed{{...}}$$ without extra words."
    
    for i in range(5):
        agent = LLMAgentBase(['answer'], f'CoT Agent {i+1}')
        answer = agent([taskInfo], cot_instruction.format(
            problem=taskInfo.content,
            context=""
        ))
        answers.append(answer[0])
    
    # Vote for most consistent
    candidates = "\\n".join([f"Candidate {i+1}: {ans.content}" for i, ans in enumerate(answers)])
    vote_instruction = "Given the diverse nature of the Combined dataset, which includes tasks ranging from commonsense reasoning and reading comprehension to mathematical problem-solving and code generation, we need a robust self-consistency voting prompt that can effectively handle the varying output formats and evaluation metrics. The key challenges are: (1) diverse output formats (text, numerical answers, code snippets), (2) varying levels of reasoning complexity, and (3) the need to consistently identify the most reliable answer across different candidate solutions.\\n\\nTo address these challenges, we refine the self-consistency voting prompt to explicitly consider the type of question and the expected output format. We also add instructions to prioritize answers that demonstrate coherent reasoning and consistency with known facts or mathematical principles. For mathematical problems, we emphasize the importance of verifying the correctness of the final numerical answer. For code generation tasks, we prioritize solutions that compile and produce the correct output for a given set of test cases.\\n\\nVote the most consistent final answer among the following candidates, considering the type of question and the expected output format. Prioritize answers that demonstrate coherent reasoning and consistency with known facts or mathematical principles. For mathematical problems, verify the correctness of the final numerical answer. For code generation tasks, prioritize solutions that compile and produce the correct output.\\n\\n{candidates}\\n\\n<response>\\n<voted><![CDATA[\\nYour chosen answer.\\n]]></voted>\\n</response>\\n\\nAdditionally, if the question is multiple-choice with options A/B/C/D, END your overall output with exactly one line: The final answer is: X."
    
    voter = LLMAgentBase(['voted'], 'Voting Agent')
    voted = voter([taskInfo], vote_instruction.format(candidates=candidates))
    return voted[0].content if hasattr(voted[0], 'content') else voted[0]
"""
}

SelfRefine = {
    "thought": "Generate via CoT and iteratively refine up to five iterations.",
    "name": "SelfRefine",
    "code": """def forward(self, taskInfo):
    # Initial solution
    cot_instruction = "You are a careful reasoner. Think step by step and derive a concise final answer.\\n\\nProblem:\\n{problem}\\n\\nContext:\\n{context}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).\\n - If the problem is mathematics, also append a final line with the canonical math format: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<reasoning><![CDATA[\\nYour step-by-step reasoning here.\\n]]></reasoning>\\n<final_answer><![CDATA[\\nYour final answer here.\\n]]></final_answer>\\n</response>"
    
    initial_agent = LLMAgentBase(['reasoning', 'final_answer'], 'Initial Agent')
    reasoning, solution = initial_agent([taskInfo], cot_instruction.format(
        problem=taskInfo.content,
        context=""
    ))
    current_solution = solution.content if hasattr(solution, 'content') else solution
    
    # Refine up to 5 iterations
    refine_instruction = "You will iteratively refine a solution up to five iterations.\\n\\nProblem:\\n{problem}\\n\\nCurrent Solution:\\n{solution}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).\\n - If the problem is mathematics, also append a final line with the canonical math format: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<notes><![CDATA[\\nYour analysis of the current solution and improvement suggestions.\\n]]></notes>\\n<refined><![CDATA[\\nYour improved solution.\\n]]></refined>\\n</response>"
    
    for iteration in range(5):
        refiner = LLMAgentBase(['notes', 'refined'], f'Refiner Agent {iteration+1}')
        notes, refined = refiner([taskInfo], refine_instruction.format(
            problem=taskInfo.content,
            solution=current_solution
        ))
        current_solution = refined.content if hasattr(refined, 'content') else refined
    
    return current_solution
"""
}

Ensemble = {
    "thought": "Three LLM agents answer, then aggregate via pairwise ranking into a final solution.",
    "name": "Ensemble",
    "code": """def forward(self, taskInfo):
    # Three agents answer
    agents = []
    member_instruction = "Act as agent {source}. Answer succinctly.\\n\\nProblem:\\n{problem}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).\\n - If the problem is mathematics, also append a final line with the canonical math format: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<answer><![CDATA[\\nYour answer here.\\n]]></answer>\\n</response>"
    
    for i in range(3):
        agent = LLMAgentBase(['answer'], f'Agent {i+1}')
        answer = agent([taskInfo], member_instruction.format(source=f'Agent {i+1}', problem=taskInfo.content))
        if answer and len(answer) > 0:
            agents.append(answer[0])
    else:
            # Fallback if agent fails
            fallback_agent = LLMAgentBase(['answer'], f'Fallback Agent {i+1}')
            fallback_answer = fallback_agent([taskInfo], "Provide a simple answer to the problem.")
            agents.append(fallback_answer[0] if fallback_answer else Info('answer', f'Agent {i+1}', 'No answer generated', -1))
    
    # Pairwise ranking
    answers = "\\n".join([f"Answer {i+1}: {ans.content}" for i, ans in enumerate(agents)])
    rank_instruction = "Given the following answers from different agents, rank them and select the best one.\\n\\nAnswers:\\n{answers}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).\\n - If the problem is mathematics, also append a final line with the canonical math format: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<final><![CDATA[\\nYour selected answer.\\n]]></final>\\n</response>"
    
    ranker = LLMAgentBase(['final'], 'Ranking Agent')
    final = ranker([taskInfo], rank_instruction.format(answers=answers))
    return final[0].content if hasattr(final[0], 'content') else final[0]
"""
}

Testing = {
    "thought": "Generate test cases for produced code given the problem and current solution.",
    "name": "Testing",
    "code": """def forward(self, taskInfo):
    # First generate a solution
    cot_instruction = "You are a careful reasoner. Think step by step and derive a concise final answer.\\n\\nProblem:\\n{problem}\\n\\nContext:\\n{context}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).\\n - If the problem is mathematics, also append a final line with the canonical math format: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<reasoning><![CDATA[\\nYour step-by-step reasoning here.\\n]]></reasoning>\\n<final_answer><![CDATA[\\nYour final answer here.\\n]]></final_answer>\\n</response>"
    
    solver = LLMAgentBase(['reasoning', 'final_answer'], 'Solver Agent')
    reasoning, solution = solver([taskInfo], cot_instruction.format(
        problem=taskInfo.content,
        context=""
    ))
    current_solution = solution.content if hasattr(solution, 'content') else solution
    
    # Generate tests
    test_instruction = "Generate test cases for the given solution.\\n\\nProblem:\\n{problem}\\n\\nSolution:\\n{solution}\\n\\nCreate comprehensive test cases that validate the solution.\\n\\n<response>\\n<tests><![CDATA[\\nYour test cases here.\\n]]></tests>\\n</response>"
    
    tester = LLMAgentBase(['tests'], 'Testing Agent')
    tests = tester([taskInfo], test_instruction.format(
        problem=taskInfo.content,
        solution=current_solution
    ))
    
    # Return solution with tests
    return f"Solution: {current_solution}\\n\\nTests: {tests[0].content}"
"""
}

EarlyExit = {
    "thought": "Early exit operator that terminates architecture sampling when appropriate.",
    "name": "EarlyExit",
    "code": """def forward(self, taskInfo):
    exit_instruction = "If the answer is already determined or unsolvable, signal early exit with a reason.\\n\\nProblem:\\n{problem}\\n\\n<response>\\n<reason><![CDATA[\\nYour reasoning for early exit or continuation.\\n]]></reason>\\n</response>"
    
    exit_checker = LLMAgentBase(['reason'], 'Early Exit Agent')
    reason = exit_checker([taskInfo], exit_instruction.format(problem=taskInfo.content))
    if reason[0].content and "exit" in reason[0].content.lower():
        return "EARLY_EXIT"
    
    # If not exiting, proceed with normal CoT
    cot_instruction = "You are a careful reasoner. Think step by step and derive a concise final answer.\\n\\nProblem:\\n{problem}\\n\\nContext:\\n{context}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).\\n - If the problem is mathematics, also append a final line with the canonical math format: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<reasoning><![CDATA[\\nYour step-by-step reasoning here.\\n]]></reasoning>\\n<final_answer><![CDATA[\\nYour final answer here.\\n]]></final_answer>\\n</response>"
    
    cot_agent = LLMAgentBase(['reasoning', 'final_answer'], 'CoT Agent')
    reasoning, final_answer = cot_agent([taskInfo], cot_instruction.format(
        problem=taskInfo.content,
        context=""
    ))
    return final_answer.content if hasattr(final_answer, 'content') else final_answer
"""
}

Reflexion = {
    "thought": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution.",
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning
    cot_initial_instruction = "You are a careful reasoner. Think step by step and derive a concise final answer.\\n\\nProblem:\\n{problem}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).\\n - If the problem is mathematics, also append a final line with the canonical math format: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<reasoning><![CDATA[\\nYour step-by-step reasoning here.\\n]]></reasoning>\\n<final_answer><![CDATA[\\nYour final answer here.\\n]]></final_answer>\\n</response>"

    # Instruction for reflecting on previous attempts and feedback to improve
    cot_reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better.\\n\\nProblem:\\n{problem}\\n\\nStrict answer policy:\\n - Output EXACTLY the answer string with no prefixes/suffixes.\\n - Do not include quotes, year, citations, or extra words.\\n - For titles, return the title text only. For numbers, return only the number.\\n - If the question is multiple-choice with options labeled A/B/C/D, END your overall output with exactly one line: The final answer is: X (where X is A, B, C, or D).\\n - If the problem is mathematics, also append a final line with the canonical math format: $$\\\\boxed{{<final answer only>}}$$.\\n - If the problem is a coding task (HumanEval-style), output ONLY a Python fenced code block implementing the required function, with no extra prose. If the problem states a required function name, define EXACTLY that name (match case and spelling) in the code block. Begin with: def <that_exact_name>(...):\\n\\n<response>\\n<reasoning><![CDATA[\\nYour step-by-step reasoning here.\\n]]></reasoning>\\n<final_answer><![CDATA[\\nYour final answer here.\\n]]></final_answer>\\n</response>"
    
    cot_agent = LLMAgentBase(['reasoning', 'final_answer'], 'Chain-of-Thought Agent')

    # Instruction for providing feedback and correcting the answer
    critic_instruction = "Please review the answer above and criticize where it might be wrong. Consider the task type (MMLU multiple-choice, MATH problem, or HumanEval code generation) and ensure the output format is correct. If you are absolutely sure it is correct, output 'True' in 'correct'.\\n\\n<response>\\n<feedback><![CDATA[\\nYour feedback here.\\n]]></feedback>\\n<correct><![CDATA[\\nTrue or False\\n]]></correct>\\n</response>"
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent')
    
    N_max = 5 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    reasoning, final_answer = cot_agent(cot_inputs, cot_initial_instruction.format(problem=taskInfo.content), 0)

    for i in range(N_max):
        # Get feedback and correct status from the critic
        feedback, correct = critic_agent([taskInfo, reasoning, final_answer], critic_instruction, i)
        if correct.content and "true" in correct.content.lower():
            break
            
        # Add feedback to the inputs for the next iteration
        cot_inputs.extend([reasoning, final_answer, feedback])

        # Reflect on previous attempts and refine the answer
        reasoning, final_answer = cot_agent(cot_inputs, cot_reflect_instruction.format(problem=taskInfo.content), i + 1)
    
    return final_answer.content if hasattr(final_answer, 'content') else final_answer
"""
}

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on the Combined benchmark, which includes MMLU (multiple-choice questions), MATH (mathematical problems), and HumanEval (code generation).

## An example question from the Combined benchmark:

Answer the following multiple choice question.

The constellation ... is a bright W-shaped constellation in the northern sky.

(A) Centaurus
(B) Cygnus
(C) Cassiopeia
(D) Cepheus

# The utility code:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
client = openai.OpenAI()

# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for LLM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\n"

# Description of the role for the LLM
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    \"""
    Function to get JSON response from GPT model.
    
    Args:
    - msg (str): The user message.
    - model (str): The model to use.
    - system_message (str): The system message.
    - temperature (float): Sampling temperature.
    
    Returns:
    - dict: The JSON response.
    \"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase:
    \"""
    Base class for an LLM agent.
    
    Attributes:
    - output_fields (list): Fields expected in the output.
    - agent_name (str): Name of the agent.
    - role (str): Role description for the agent.
    - model (str): Model to be used. (option. Keep it default.)
    - temperature (float): Sampling temperature.
    - id (str): Unique identifier for the agent instance.
    \"""

    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()
    
    def generate_prompt(self, input_infos, instruction) -> str:
        \"""
        Generates a prompt for the LLM.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        
        Returns:
        - tuple: System prompt and user prompt.

        An example of a generated prompt:
        ""
        You are a helpful assistant.
        
        # Output Format:
        Reply EXACTLY with the following JSON format.
        ...

        # Your Task:
        You will be given some number of paired example inputs and outputs. The outputs ...

        ### thinking #1 by Chain-of-Thought Agent hkFo (yourself):
        ...
        
        ### code #1 by Chain-of-Thought Agent hkFo (yourself):
        ...

        ### answer by Chain-of-Thought Agent hkFo's code evaluator:...


        # Instruction: 
        Please think step by step and then solve the task by writing the code.
        ""
        \"""
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt 

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        \"""
        Queries the LLM with provided input information and instruction.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        - iteration_idx (int): Iteration index for the task.
        
        Returns:
        - output_infos (list[Info]): Output information.
        \"""
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"
    
    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        # Note:
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    \"""
    Fill in your code here.
    \"""
    def forward(self, taskInfo) -> Union[Info, str]:
        \"""
        Placeholder method for processing task information.
        
        Args:
        - taskInfo (Info): Task information.
        
        Returns:
        - Answer (Union[Info, str]): Your FINAL Answer. Return either a namedtuple Info or a string of answers.
        \"""
        pass
```
# Discovered architecture archive
Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the correct rate on a validation question set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture. 
Finally, the last key ("code") corresponds to the exact "forward()" function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. 
Also, it could be helpful to set the LLM's role and temperature to further control the LLM's response. Note that the LLMAgentBase() will automatically parse the output and return a list of "Infos". You can get the content by Infos.content. 
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.

## WRONG Implementation examples:
Here are some mistakes you may make:

1. This is WRONG: ```
feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
```
It is wrong to use "Info('feedback', 'Critic Agent', thinking, 0)". The returned "feedback" from LLMAgentBase is already Info.

2. This is WRONG: ```
# Debugging: Log the generated answer
print('Generated Answer:', ...)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'
```
First, the len(feedback_info) will not work.
Second, you should never return an error message. You should always return the best answer you can get.
Third, you should never print anything in the code.
Lastly, again, DO NOT CREATE Info object by yourself.

3. This is WRONG: ```
all_thinking = []
all_answers = []
for agent, role in zip(agents, roles):
    outputs = agent([taskInfo], independent_reasoning_instruction.format(role=role))
    all_thinking.append(outputs[0].content)
    all_answers.append(outputs[1].content)

# Aggregate the reasoning paths and answers
aggregated_thinking = '\n'.join(all_thinking)
aggregated_answers = '\n'.join(all_answers)
```
You SHOULD NOT extract the content from the Info object by yourself. You should use the Info object directly. If you want to aggregate the content, you should just put those Info objects into a list and then use the list as input to the next LLM agent.

4. This is WRONG: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
response_infos = reasoning_agent([taskInfo] + ..., reasoning_instruction)
    
# Extract the final answer from the response_infos
for info in response_infos:
    if info.name == 'final_answer':
        return info
# Fallback if no answer is found
return Info('answer', 'Final Decision Agent', 'No answer generated.', 0)
```
You should not extract the final answer by yourself. You SHOULD directly return the answer Info. Also, you should always return the best answer you can get.
CORRECT example: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = reasoning_agent([taskInfo] + ..., reasoning_instruction)
return answer
```

5. This is WRONG: ```
import wikipedia
import requests
import openai
```
**CRITICAL: DO NOT USE EXTERNAL DEPENDENCIES!** The system only has access to basic Python modules. DO NOT import:
- wikipedia, requests, openai, or any external APIs
- Any modules that require internet access
- Any third-party libraries not in the basic Python environment

Only use these allowed imports: json, re, random, math, collections, typing, numpy, string, itertools, functools, datetime, copy, sys, os, traceback

# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to maximize "fitness" by proposing interestingly new agents. 
Observe the discovered architectures carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative to think about the next interesting architecture to try. You are encouraged to draw inspiration from related LLM agent papers or academic papers from other research areas.

**CRITICAL DIVERSITY REQUIREMENT**: 
- DO NOT generate another testing/refinement based approach if recent generations have focused on this pattern
- Explore DIFFERENT architectural paradigms: reasoning strategies, ensemble methods, specialized agents, etc.
- If the archive shows similar approaches (like multiple testing-based agents), propose a COMPLETELY DIFFERENT direction
- Consider: multi-agent collaboration, specialized reasoning, different prompting strategies, or novel control flows

Using the knowledge learned from the archive and the inspiration from academic literature to give the next interesting architecture.
THINK OUTSIDE THE BOX.
"""

Reflexion_prompt_1 = f""""[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:"

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings. 
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""

Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
"""

def get_init_archive():
    return [CoT, Debate, SelfConsistency, SelfRefine, Ensemble, Testing, EarlyExit, Reflexion]

def get_prompt(current_archive, adaptive=False):
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))
    return system_prompt, prompt

def get_reflexion_prompt(prev_example):
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    r1 = Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str) if prev_example else Reflexion_prompt_1.replace("[EXAMPLE]", "")
    return r1, Reflexion_prompt_2
