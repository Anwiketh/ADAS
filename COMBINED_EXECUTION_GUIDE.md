# ADAS Combined Benchmark: Complete Training & Testing Guide

## Overview

The ADAS (Automated Design of Agentic Systems) Combined benchmark uses **evolutionary search** to automatically design agent architectures. Unlike traditional ML training, there are no neural network weights to save/load. Instead, the system discovers and evolves agent designs through iterative refinement.

## How the System Works

1. **Meta Agent Search**: A "meta" agent (Gemini 2.0 Flash) designs new agent architectures in code
2. **Evaluation**: Each designed agent is tested on the combined dataset
3. **Evolution**: Successful designs inform future iterations
4. **Archive**: Best performing agents are saved in JSON format

## Key Components

### 1. Dataset Structure
The combined dataset (`combined_train.jsonl`, `combined_test.jsonl`) contains three task types:
- **MMLU**: Multiple choice questions (A/B/C/D)
- **Math**: Mathematical word problems
- **HumanEval**: Python code generation tasks

### 2. Core Files

#### `Combined/search.py` - Main Execution Script
```python
# Key command line arguments (lines 271-280)
parser.add_argument("--train_data", type=str, default="combined_train.jsonl")
parser.add_argument("--test_data", type=str, default="combined_test.jsonl")
parser.add_argument("--valid_size", type=int, default=100)
parser.add_argument("--max_workers", type=int, default=16)
parser.add_argument("--save_dir", type=str, default="results/")
parser.add_argument("--expr_name", type=str, default="combined_gemini")
parser.add_argument("--n_generation", type=int, default=20)
parser.add_argument("--debug_max", type=int, default=2)
parser.add_argument("--model", type=str, default="gemini-2.0-flash")
parser.add_argument("--api_key", type=str, required=True)
```

#### `Combined/utils.py` - Advanced Answer Parsing Logic
```python
# Advanced parsing functions identical to combined.py

def calculate_mmlu_score(expected_output, prediction):
    """Calculate MMLU score using regex pattern matching"""
    prediction = str(prediction)
    match = re.search(r"The final answer is: ([A-Z])", prediction)
    if match:
        extracted_answer = match.group(1)
        predicted_index = ord(extracted_answer) - ord("A")
        if predicted_index == expected_output:
            return 1.0, extracted_answer
    return 0.0, ""

def extract_model_answer(text):
    """Extract model answer using XML tags, boxed expressions, or last sentence"""
    # Prefer XML-like tags
    tag_patterns = [
        r"<final_answer>\s*([\s\S]*?)\s*</final_answer>",
        r"<answer>\s*([\s\S]*?)\s*</answer>",
        # ... more patterns
    ]
    # Then prefer last boxed expression
    # Fallback to last sentence

def calculate_math_score(expected_output, prediction):
    """Calculate math score using symbolic equality"""
    expected_answer = extract_model_answer(expected_output)
    predicted_answer = extract_model_answer(prediction)
    return (1 if math_equal(predicted_answer, expected_answer) else 0, predicted_answer)

def check_humaneval_solution(solution, test, entry_point):
    """Execute and test HumanEval solutions with timeout"""
    # Sanitize code, execute with timeout, run tests
```

#### `Combined/combined_prompt.py` - Agent Design Prompts
```python
# Lines 115-120: Task type definitions
You are designing agentic systems to perform well on a COMBINED benchmark that mixes:
- MMLU multiple-choice questions (return exactly one of A/B/C/D),
- MATH problems (return only the final answer, numeric or simplified fraction/word),
- HumanEval code generation tasks (return only valid Python code defining the requested function).
```

## Key Parsing Logic

### 1. LLM Response Parsing (Lines 240-267 in `Combined/search.py`)

```python
def evaluate_forward_fn(args, forward_str):
    # ... setup code ...
    
    for i, res in enumerate(results):
        pred = res.content if isinstance(res, Info) else res
        ex = examples[i]
        correct = 0
        
        # Extract string from various output formats
        pred = extract_string_from_output(pred)
        
        # MMLU parsing - uses exact logic from combined.py
        if ex["type"] == "mmlu":
            score, _ = calculate_mmlu_score(ex["answer"], pred)
            correct = int(score)
            
        # Math parsing - uses exact logic from combined.py
        elif ex["type"] == "math":
            score, _ = calculate_math_score(ex["solution"], pred)
            correct = score
            
        # HumanEval parsing - uses exact logic from combined.py
        elif ex["type"] == "humaneval":
            try:
                ret = check_humaneval_solution(pred, ex.get("test", ""), ex.get("entry_point", ""))
                if isinstance(ret, (list, tuple)) and len(ret) >= 2:
                    correct = 1 if ret[0] == "PASS" else 0
                else:
                    # Fallback: check if it produces valid, non-empty code
                    if isinstance(pred, str) and "def " in pred and len(pred) > 20:
                        correct = 1
            except Exception as e:
                print(f"HumanEval evaluation error: {e}")
                # Fallback: check if it produces valid, non-empty code
                if isinstance(pred, str) and "def " in pred and len(pred) > 20:
                    correct = 1
        
        acc_list.append(correct)
```

### 2. Individual Dataset Parsing

#### MMLU/GPQA Parsing (Lines 320-359 in `_mmlu/search.py`)
```python
for q_idx, res in enumerate(results):
    try:
        if isinstance(res, str) and res in LETTER_TO_INDEX:
            predicted_idx = LETTER_TO_INDEX[res]
        elif 'A)' in res:
            predicted_idx = 0
        elif 'B)' in res:
            predicted_idx = 1
        elif 'C)' in res:
            predicted_idx = 2
        elif 'D)' in res:
            predicted_idx = 3
        elif isinstance(res, list):
            try_res = res[1]
            predicted_idx = LETTER_TO_INDEX[try_res.content]
        elif res.content in LETTER_TO_INDEX:
            predicted_idx = LETTER_TO_INDEX[res.content]
        # ... more parsing logic
        
        if predicted_idx == answers[q_idx]:
            acc_list.append(1)
        else:
            acc_list.append(0)
```

#### Math Parsing (Lines 8-15 in `_transfer_math/gsm8k_utils.py`)
```python
def score_gsm8k(target: str, prediction: str) -> bool:
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")
    
    target = target.replace(",", "")
    prediction = prediction.replace(",", "")
    
    return target == prediction

def extract_answer_str(answer_str):
    pattern = r'#### (-?\d+)'
    match = re.search(pattern, answer_str)
    if match:
        answer = match.group(1)
    else:
        raise AssertionError("No match found")
    return answer
```

## Execution Commands

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
cp env.example .env
# Edit .env file and add your actual Gemini API key
```

### 2. Configure API Key
Create a `.env` file in the project root with your Gemini API key:

```bash
# .env file
GOOGLE_API_KEY=your-actual-gemini-api-key-here
```

**Get your API key from:** https://makersuite.google.com/app/apikey

### 3. Basic Training & Testing
```bash
cd Combined
python search.py
```

### 4. Customized Training
```bash
cd Combined
python search.py \
    --n_generation 10 \
    --valid_size 50 \
    --max_workers 8 \
    --expr_name "my_experiment" \
    --model "gemini-2.0-flash"
```

### 5. Individual Dataset Training
```bash
# MMLU
cd _mmlu
python search.py

# GPQA
cd _gpqa
python search.py

# Math (GSM8K)
cd _transfer_math
python evaluation_gsm8k.py
```

## Training Process Flow

### Phase 1: Initial Evaluation (Lines 116-140 in `Combined/search.py`)
```python
def search(args):
    # Load or initialize archive
    archive = get_init_archive()
    
    # Evaluate initial solutions
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
```

### Phase 2: Evolutionary Search (Lines 142-190 in `Combined/search.py`)
```python
for n in range(start_gen, args.n_generation):
    print(f"============ Generation {n + 1} ============")
    
    # Generate new solution using meta agent
    sys, prompt = get_prompt(archive)
    msg_list = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
    next_sol = get_json_response_from_gemini_reflect(msg_list, args.model)
    
    # Reflexion refinement
    p1, p2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
    # ... reflexion steps
    
    # Evaluate new solution
    acc = []
    for _ in range(args.debug_max):
        try:
            acc = evaluate_forward_fn(args, next_sol["code"])
            break
        except Exception as e:
            print("During evaluation:", e)
            continue
    
    # Save to archive
    next_sol['fitness'] = bootstrap_confidence_interval(acc)
    next_sol['generation'] = n + 1
    archive.append(next_sol)
```

### Phase 3: Final Testing (Lines 196-219 in `Combined/search.py`)
```python
def evaluate(args):
    # Load archive
    with open(archive_path, "r") as f:
        archive = json.load(f)
    
    # Test each solution on test set
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
```

## Output Files

### Training Results
- `results/combined_gemini_archive.json`: Archive of all discovered agents
- `results/combined_gemini_evaluate.json`: Test results for all agents

### Individual Dataset Results
- `results/mmlu_gemini_results.json`
- `results/gpqa_gemini_results.json`
- `results/mgsm_gemini_results.json`
- etc.

## Key Differences from Traditional ML

1. **No Weights**: Agents are code designs, not neural network weights
2. **Evolutionary**: Uses meta-agent to design new architectures
3. **Multi-modal**: Handles multiple task types in unified framework
4. **Reflexion**: Agents can reflect and improve their own designs

## Advanced Parsing Features

The system now uses **identical parsing logic** to `combined.py` for all task types:

### MMLU Parsing
- **Pattern Matching**: Extracts answers using regex `"The final answer is: ([A-Z])"`
- **Index Conversion**: Maps A/B/C/D to 0/1/2/3 indices
- **Robust Extraction**: Handles various response formats

### Math Parsing
- **XML Tag Extraction**: Prioritizes `<final_answer>`, `<answer>`, `<refined>` tags
- **LaTeX Boxed Expressions**: Extracts `\boxed{...}` content
- **Symbolic Equality**: Uses SymPy for mathematical expression comparison
- **Numerical Tolerance**: Handles floating-point precision with `abs_tol=1e-3`
- **Percentage Handling**: Automatically converts percentages to decimals

### HumanEval Parsing
- **Code Execution**: Actually runs generated Python code
- **Test Execution**: Executes unit tests against generated functions
- **Timeout Protection**: 15-second timeout to prevent infinite loops
- **Error Handling**: Comprehensive error capture and reporting
- **Helper Injection**: Automatically adds required helper functions for specific problems

## Gemini API Integration

### Model Options
- **`gemini-2.0-flash`**: Latest and fastest model (default)
- **`gemini-1.5-flash`**: Previous generation, good performance
- **`gemini-1.5-pro`**: Most capable but slower

### API Setup
```python
import google.generativeai as genai

# Configure API
genai.configure(api_key="your-api-key")

# Create model instance
client = genai.GenerativeModel("gemini-2.0-flash")

# Generate content
response = client.generate_content(
    "Your prompt here",
    generation_config=genai.types.GenerationConfig(
        temperature=0.5,
        max_output_tokens=4096,
    )
)
```

### JSON Response Handling
Since Gemini doesn't have built-in JSON response formatting like OpenAI, the system includes JSON extraction logic:
```python
def get_json_response_from_gemini(msg, model, system_message, temperature=0.5):
    response = client.generate_content(
        f"{system_message}\n\n{msg}",
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=4096,
        )
    )
    content = response.text
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
        return {"thinking": content, "answer": content}
```

## Troubleshooting

### Common Issues
1. **API Key**: Ensure `GOOGLE_API_KEY` is set in your `.env` file
2. **Dataset**: Verify `combined_train.jsonl` and `combined_test.jsonl` exist
3. **Memory**: Reduce `max_workers` if running out of memory
4. **Rate Limits**: System includes backoff retry logic
5. **JSON Parsing**: Gemini responses may need manual JSON extraction

### Debug Mode
```bash
python search.py --debug_max 1 --n_generation 2
```

## Performance Metrics

The system reports:
- **Fitness**: Bootstrap confidence interval of accuracy on validation set
- **Test Fitness**: Bootstrap confidence interval of accuracy on test set
- **Generation**: Which evolutionary generation produced the agent

Example output:
```
Accuracy: 95% Bootstrap CI: (25.4%, 31.6%), Median: 28.4%
```

## Migration from OpenAI

### Key Changes Made
1. **Import**: `import openai` → `import google.generativeai as genai`
2. **Client**: `openai.OpenAI()` → `genai.GenerativeModel()`
3. **API Call**: `client.chat.completions.create()` → `client.generate_content()`
4. **Model Names**: `gpt-4o` → `gemini-2.0-flash`
5. **Response Format**: Added JSON extraction logic for Gemini responses
6. **Error Handling**: Updated exception types and retry logic
