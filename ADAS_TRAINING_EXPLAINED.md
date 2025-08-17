# ADAS Training System: Complete Beginner's Guide

## What is ADAS?

**ADAS** stands for **Automated Design of Agentic Systems**. Think of it like an AI that designs other AIs! 

### The Big Picture Analogy

Imagine you're trying to build the perfect chef. Instead of teaching one chef to cook everything, you have a **master chef** (the meta-agent) who designs different cooking strategies and tests them to see which ones work best. The master chef keeps improving the recipes based on what works and what doesn't.

## How Does Training Work? Step-by-Step

### 1. The Problem We're Solving

We have a **Combined Benchmark** - a test that includes three types of problems:
- **MMLU**: Multiple choice questions (like a test with A, B, C, D options)
- **Math**: Math word problems (like "If John has 5 apples and gives 2 to Mary...")
- **HumanEval**: Programming problems (like "Write a function that sorts a list")

### 2. The Initial "Chefs" (Agent Operators)

We start with 7 different "cooking strategies" (agent operators):

#### 🧠 **CoT (Chain-of-Thought)**
- **What it does**: Thinks step by step before answering
- **Analogy**: Like a student who writes out their work before giving the final answer
- **Example**: "First, I need to understand the question. Then, I'll break it down into steps..."

#### 🗣️ **Debate**
- **What it does**: Three AI agents argue about the answer, then vote
- **Analogy**: Like having three experts discuss a problem and come to a consensus
- **Example**: Expert 1 says "A", Expert 2 says "B", Expert 3 says "A" → Final answer: "A"

#### 🔄 **SelfConsistency**
- **What it does**: Tries the same problem 5 times and picks the most common answer
- **Analogy**: Like flipping a coin multiple times to make sure you get the right result
- **Example**: Tries 5 times, gets A, A, B, A, A → Final answer: "A"

#### ✨ **SelfRefine**
- **What it does**: Makes an answer, then keeps improving it up to 5 times
- **Analogy**: Like writing an essay, then editing it multiple times to make it better
- **Example**: First draft → Edit → Edit → Edit → Final version

#### 👥 **Ensemble**
- **What it does**: Three different agents answer, then rank their answers
- **Analogy**: Like having three different doctors diagnose the same patient
- **Example**: Doctor 1, 2, and 3 give diagnoses → Rank them by confidence

#### 🧪 **Testing**
- **What it does**: Generates test cases to check if the answer is correct
- **Analogy**: Like a scientist who creates experiments to verify their hypothesis
- **Example**: "If my answer is right, then X should equal Y. Let me test that."

#### 🚪 **EarlyExit**
- **What it does**: Checks if the problem is too easy or impossible, then exits early
- **Analogy**: Like a detective who quickly realizes a case is solved or unsolvable
- **Example**: "This is obviously A, no need to think further" or "This is impossible to solve"

### 3. The Training Process (Evolutionary Search)

#### Phase 1: Initial Evaluation
```
Step 1: Test each "chef" on 5 problems
Step 2: Calculate their "fitness score" (how well they did)
Step 3: Save the results
```

**What happens**: Each of our 7 agents tries to solve 5 random problems from our dataset. We calculate their accuracy (how many they got right).

#### Phase 2: The Master Chef Creates New Recipes
```
Step 4: Show all results to the master chef (Gemini 2.0 Flash)
Step 5: Master chef designs a new "cooking strategy"
Step 6: Test the new strategy
Step 7: If it works, keep it; if not, try again
```

**What happens**: 
1. We show Gemini 2.0 Flash all the results from our 7 agents
2. We ask it: "Look at these strategies and their results. Can you design a better strategy?"
3. Gemini creates a new agent with a new approach
4. We test this new agent on 5 problems
5. If it gets better results, we keep it; if not, we ask Gemini to try again

#### Phase 3: Reflection and Improvement
```
Step 8: If the new strategy fails, ask Gemini to reflect and improve
Step 9: Gemini looks at what went wrong and fixes it
Step 10: Test the improved version
```

**What happens**: If the new agent doesn't work well, we show Gemini what went wrong and ask it to fix the strategy. This is like a teacher giving feedback to a student.

#### Phase 4: Repeat and Evolve
```
Step 11: Add successful new agents to our collection
Step 12: Go back to Step 4 and create more agents
Step 13: Keep going for 20 generations (or however many you specify)
```

**What happens**: We keep creating new agents, testing them, and keeping the best ones. Over time, our collection of agents gets better and better.

### 4. The Technical Details

#### How Agents Are Represented
Each agent is just **Python code** that looks like this:
```python
def forward(self, taskInfo):
    # This is where the agent's strategy goes
    # It receives a problem and returns an answer
    return "A"  # or whatever the answer is
```

#### How Problems Are Formatted
Problems come in three formats:

**MMLU (Multiple Choice)**:
```
Question: What is the capital of France?
(A) London
(B) Paris  
(C) Berlin
(D) Madrid
```

**Math**:
```
Solve the following math problem:
If John has 5 apples and gives 2 to Mary, how many does he have left?
```

**HumanEval (Programming)**:
```
Complete the following Python function:
def add_numbers(a, b):
    # Your code here
```

#### How Answers Are Evaluated

**MMLU**: Check if the agent picked the right letter (A, B, C, or D)
**Math**: Check if the agent gave the correct numerical answer
**HumanEval**: Actually run the code and see if it passes tests

#### How Fitness Is Calculated
```
Fitness = (Number of correct answers) / (Total number of problems)
Example: 3 correct out of 5 problems = 60% fitness
```

### 5. The Files and Their Roles

#### `Combined/search.py` - The Main Controller
- **Role**: Like the conductor of an orchestra
- **What it does**: 
  - Loads the dataset
  - Runs the training loop
  - Calls Gemini to create new agents
  - Saves results to files

#### `Combined/combined_prompt.py` - The Recipe Book
- **Role**: Like a cookbook with all the basic recipes
- **What it does**: 
  - Defines the 7 initial agents
  - Contains the prompts for Gemini
  - Stores the "master chef instructions"

#### `Combined/utils.py` - The Kitchen Tools
- **Role**: Like the utensils and equipment in a kitchen
- **What it does**: 
  - Formats questions for agents
  - Evaluates answers (checks if they're right)
  - Calculates fitness scores

#### `dataset/combined_train.jsonl` - The Training Problems
- **Role**: Like a stack of practice tests
- **What it does**: Contains all the problems agents practice on

#### `dataset/combined_test.jsonl` - The Final Exam
- **Role**: Like the actual test that determines final grades
- **What it does**: Contains problems for final evaluation

### 6. The Training Loop in Detail

#### Generation 0: Initial Agents
```
1. Load the 7 initial agents (CoT, Debate, SelfConsistency, etc.)
2. For each agent:
   a. Pick 5 random problems from the training set
   b. Let the agent solve them
   c. Calculate accuracy
   d. Save the result
3. Save all results to "prelim_train_archive.json"
```

#### Generation 1: First New Agent
```
1. Load all previous results
2. Create a prompt for Gemini: "Here are 7 agents and their results. Design a better one."
3. Send to Gemini and get back a new agent design
4. Test the new agent on 5 problems
5. If accuracy > 0.01 (at least one correct answer):
   a. Save the new agent
   b. Add it to the archive
6. If accuracy = 0:
   a. Ask Gemini to fix the agent
   b. Test again
   c. Repeat up to 2 times
```

#### Generation 2, 3, 4...: More Evolution
```
1. Load all previous agents and results
2. Ask Gemini to design an even better agent
3. Test and save successful ones
4. Keep going until you reach the target number of generations
```

### 7. The Output Files

#### `results/prelim_train_archive.json`
- **Contains**: All agents created during training
- **Format**: JSON with agent names, code, and fitness scores
- **Example**:
```json
[
  {
    "name": "CoT",
    "code": "def forward(self, taskInfo): ...",
    "fitness": "95% Bootstrap CI: (0.0%, 80.0%), Median: 40.0%",
    "generation": "initial"
  },
  {
    "name": "Hybrid_Agent_1", 
    "code": "def forward(self, taskInfo): ...",
    "fitness": "95% Bootstrap CI: (20.0%, 100.0%), Median: 60.0%",
    "generation": 1
  }
]
```

### 8. Key Concepts Explained

#### What is "Fitness"?
- **Definition**: How well an agent performs on the problems
- **Analogy**: Like a student's grade on a test
- **Calculation**: (Correct answers) / (Total problems)
- **Example**: 3 out of 5 correct = 60% fitness

#### What is "Bootstrap Confidence Interval"?
- **Definition**: A statistical measure of how reliable the fitness score is
- **Analogy**: Like saying "I'm 95% confident the student's true ability is between 40% and 80%"
- **Why it matters**: A score of 60% on 5 problems might not be very reliable

#### What is "Generation"?
- **Definition**: One complete cycle of creating and testing new agents
- **Analogy**: Like one round of a game where everyone gets a turn
- **Example**: Generation 0 = initial agents, Generation 1 = first new agent, etc.

#### What is "Archive"?
- **Definition**: The collection of all agents created so far
- **Analogy**: Like a library of all the recipes that have been tried
- **Purpose**: To show Gemini what's been done before so it can improve

### 9. The Command Line Arguments

When you run:
```bash
python3 search.py --n_generation 3 --valid_size 5 --debug_max 1 --expr_name "prelim_train" --max_workers 2
```

Here's what each part means:

- `--n_generation 3`: Run 3 generations (create 3 new agents)
- `--valid_size 5`: Test each agent on 5 problems
- `--debug_max 1`: Try to fix failed agents up to 1 time
- `--expr_name "prelim_train"`: Name the output files "prelim_train_archive.json"
- `--max_workers 2`: Use 2 CPU cores for parallel processing

### 10. The Complete Flow

```
START
  ↓
Load 7 initial agents (CoT, Debate, etc.)
  ↓
Test each agent on 5 problems
  ↓
Calculate fitness scores
  ↓
Save results to archive
  ↓
[LOOP: For each generation]
  ↓
Show all results to Gemini
  ↓
Ask Gemini to design a better agent
  ↓
Test the new agent
  ↓
If it works → Save it
If it fails → Ask Gemini to fix it
  ↓
Add successful agents to archive
  ↓
[END LOOP]
  ↓
Save final archive to file
  ↓
END
```

### 11. Why This Works

#### The Power of Evolution
- **Principle**: Like biological evolution, successful strategies survive and reproduce
- **In Practice**: Good agents get copied and improved, bad agents get discarded

#### The Power of Meta-Learning
- **Principle**: The master agent (Gemini) learns from the success/failure of other agents
- **In Practice**: Each generation builds on the knowledge of previous generations

#### The Power of Diversity
- **Principle**: Different strategies work for different types of problems
- **In Practice**: Having multiple approaches increases the chance of solving any given problem

### 12. Common Issues and Solutions

#### "All 0 accuracy" Error
- **What it means**: The new agent got every problem wrong
- **Why it happens**: The agent code has a bug or doesn't work
- **Solution**: Ask Gemini to fix the agent and try again

#### "Context too long" Error
- **What it means**: The prompt to Gemini is too big
- **Why it happens**: Too many agents in the archive
- **Solution**: Reduce the number of agents shown to Gemini

#### "API key not found" Error
- **What it means**: The system can't find your Google API key
- **Solution**: Set the GOOGLE_API_KEY environment variable

### 13. Advanced Concepts

#### What is "Reflexion"?
- **Definition**: When Gemini looks at its own failed attempts and fixes them
- **Analogy**: Like a student who reviews their wrong answers and learns from mistakes

#### What is "Self-Consistency"?
- **Definition**: Running the same problem multiple times and picking the most common answer
- **Analogy**: Like asking multiple people the same question and going with the majority

#### What is "Ensemble"?
- **Definition**: Combining multiple agents' answers to get a better result
- **Analogy**: Like having multiple doctors diagnose a patient and combining their opinions

### 14. The End Result

After training, you have:
1. **A collection of agents** that can solve different types of problems
2. **Performance data** showing which agents work best
3. **Evolutionary history** showing how the agents improved over time
4. **Reusable strategies** that can be applied to new problems

This is like having a team of specialized experts, each good at different types of problems, and knowing which expert to use for which type of problem.

---

## Summary

ADAS is like having a master chef who designs cooking strategies, tests them, learns from the results, and keeps improving. The training process is an evolutionary search where successful strategies survive and get improved, while unsuccessful ones are discarded. The end result is a collection of specialized AI agents that can solve different types of problems effectively.

## How to Speed Up Training Runs

### 🚀 **Speed Optimization Strategies**

#### 1. **Reduce Validation Size** (Biggest Impact)
```bash
# Slow: Tests each agent on 100 problems
python3 search.py --valid_size 100

# Fast: Tests each agent on only 5 problems  
python3 search.py --valid_size 5
```
**Why it works**: Each agent evaluation is the slowest part. Fewer problems = faster evaluation.
**Trade-off**: Less reliable fitness scores, but much faster training.

#### 2. **Increase Parallel Processing**
```bash
# Slow: Uses only 1 CPU core
python3 search.py --max_workers 1

# Fast: Uses 8 CPU cores (or however many you have)
python3 search.py --max_workers 8
```
**Why it works**: Multiple agents can be evaluated simultaneously.
**Trade-off**: Uses more CPU resources, but significantly faster.

#### 3. **Reduce Debug Attempts**
```bash
# Slow: Tries to fix failed agents up to 3 times
python3 search.py --debug_max 3

# Fast: Only tries to fix failed agents once
python3 search.py --debug_max 1
```
**Why it works**: Fewer attempts to fix broken agents = less time spent on failures.
**Trade-off**: Might miss some agents that could be fixed.

#### 4. **Use Faster Models** (If Available)
```bash
# Current: Uses Gemini 2.0 Flash (fast)
python3 search.py --model gemini-2.0-flash

# Alternative: Could use even faster models if available
python3 search.py --model gemini-1.5-flash
```
**Why it works**: Different models have different response times.
**Trade-off**: Faster models might be less capable.

#### 5. **Reduce Number of Generations** (For Testing)
```bash
# Full training: 20 generations
python3 search.py --n_generation 20

# Quick test: 3 generations
python3 search.py --n_generation 3
```
**Why it works**: Fewer generations = less total time.
**Trade-off**: Less evolution, but good for testing the system.

### ⚡ **Recommended Fast Training Commands**

#### **Ultra-Fast Test Run** (For debugging)
```bash
python3 search.py \
  --n_generation 2 \
  --valid_size 3 \
  --debug_max 1 \
  --max_workers 4 \
  --expr_name "ultra_fast_test"
```
**Time**: ~5-10 minutes
**Use case**: Testing if the system works

#### **Quick Training Run** (For development)
```bash
python3 search.py \
  --n_generation 5 \
  --valid_size 10 \
  --debug_max 1 \
  --max_workers 8 \
  --expr_name "quick_train"
```
**Time**: ~30-60 minutes
**Use case**: Developing new features

#### **Balanced Training Run** (For research)
```bash
python3 search.py \
  --n_generation 10 \
  --valid_size 20 \
  --debug_max 2 \
  --max_workers 8 \
  --expr_name "balanced_train"
```
**Time**: ~2-4 hours
**Use case**: Getting meaningful results

#### **Full Training Run** (For production)
```bash
python3 search.py \
  --n_generation 20 \
  --valid_size 50 \
  --debug_max 2 \
  --max_workers 16 \
  --expr_name "full_train"
```
**Time**: ~8-16 hours
**Use case**: Final results for papers

### 🔧 **Advanced Speed Optimizations**

#### **1. Use a More Powerful Machine**
- **More CPU cores** = higher `--max_workers`
- **Faster internet** = quicker API calls to Gemini
- **SSD storage** = faster file I/O

#### **2. Optimize Network Settings**
```bash
# If you have connection issues, reduce timeout
export GOOGLE_API_TIMEOUT=30
```

#### **3. Use Caching** (If implemented)
- Some systems cache API responses
- Reduces repeated calls to the same model

#### **4. Batch Processing** (Advanced)
- Process multiple agents in parallel
- Requires code modifications

### 📊 **Speed vs. Quality Trade-offs**

| Setting | Speed | Quality | Use Case |
|---------|-------|---------|----------|
| `--valid_size 3` | ⚡⚡⚡ | ⭐ | Testing |
| `--valid_size 10` | ⚡⚡ | ⭐⭐ | Development |
| `--valid_size 20` | ⚡ | ⭐⭐⭐ | Research |
| `--valid_size 50` | 🐌 | ⭐⭐⭐⭐ | Production |

| Setting | Speed | Quality | Use Case |
|---------|-------|---------|----------|
| `--max_workers 1` | 🐌 | ⭐⭐⭐⭐ | Debugging |
| `--max_workers 4` | ⚡ | ⭐⭐⭐ | Development |
| `--max_workers 8` | ⚡⚡ | ⭐⭐ | Research |
| `--max_workers 16` | ⚡⚡⚡ | ⭐ | Production |

### 🎯 **Pro Tips for Speed**

#### **1. Start Small, Scale Up**
```bash
# First: Test with minimal settings
python3 search.py --n_generation 1 --valid_size 3

# Then: Scale up if it works
python3 search.py --n_generation 5 --valid_size 10

# Finally: Full run
python3 search.py --n_generation 20 --valid_size 50
```

#### **2. Monitor Progress**
- Watch the progress bars to see which agents are slow
- Some agents (like Debate) are naturally slower than others
- Consider skipping very slow agents in early testing

#### **3. Use Different Machines**
- **Development**: Use your local machine with fast settings
- **Production**: Use a cloud server with more resources

#### **4. Parallel Training**
- Run multiple training sessions on different machines
- Each with different random seeds
- Combine results later

### 🚨 **Common Speed Bottlenecks**

#### **1. API Rate Limits**
- **Problem**: Gemini API has rate limits
- **Solution**: Use `--max_workers` that doesn't exceed limits
- **Typical limit**: 60 requests per minute

#### **2. Slow Agents**
- **Problem**: Some agents (Debate, Ensemble) are naturally slow
- **Solution**: Use simpler agents for speed testing
- **Fast agents**: CoT, SelfConsistency
- **Slow agents**: Debate, Ensemble, Testing

#### **3. Network Latency**
- **Problem**: Slow internet connection
- **Solution**: Use a machine with better internet
- **Alternative**: Use local models if available

#### **4. File I/O**
- **Problem**: Slow disk writes
- **Solution**: Use SSD storage
- **Alternative**: Reduce frequency of saves

### 📈 **Expected Training Times**

| Configuration | Time Estimate | Reliability |
|---------------|---------------|-------------|
| Ultra-fast test | 5-10 min | Low |
| Quick training | 30-60 min | Medium |
| Balanced training | 2-4 hours | High |
| Full training | 8-16 hours | Very High |

*Times depend on your machine, internet speed, and API response times.*

### 🎯 **Quick Start Commands**

For immediate testing:
```bash
# Test if everything works
python3 search.py --n_generation 1 --valid_size 3 --max_workers 2

# Quick development run
python3 search.py --n_generation 3 --valid_size 5 --max_workers 4

# Meaningful results
python3 search.py --n_generation 5 --valid_size 10 --max_workers 8
```

Remember: **Start small and scale up!** It's better to run a quick test and find problems early than to wait hours for a failed run.

## 🚀 **Full Training Speed Optimization**

### **For Production/Research Runs (Full Size, Maximum Speed)**

#### **Optimal Full Training Command**
```bash
python3 search.py \
  --n_generation 20 \
  --valid_size 50 \
  --debug_max 2 \
  --max_workers 16 \
  --expr_name "full_speed_optimized"
```

#### **Ultra-Optimized Full Training** (If you have powerful hardware)
```bash
python3 search.py \
  --n_generation 20 \
  --valid_size 50 \
  --debug_max 1 \
  --max_workers 32 \
  --expr_name "ultra_speed_full"
```

### **Hardware Optimization for Full Training**

#### **1. CPU Optimization**
```bash
# Check your CPU cores
nproc  # or sysctl -n hw.ncpu on Mac

# Use ALL available cores (but leave 1-2 for system)
python3 search.py --max_workers $(($(nproc) - 2))
```

#### **2. Memory Optimization**
- **Minimum**: 8GB RAM
- **Recommended**: 16GB+ RAM
- **Optimal**: 32GB+ RAM for very large runs

#### **3. Storage Optimization**
- **Use SSD**: Much faster than HDD for file I/O
- **Free space**: Ensure at least 10GB free space
- **Temp directory**: Use fast storage for temporary files

#### **4. Network Optimization**
```bash
# Test your internet speed
curl -o /dev/null -s -w "Download: %{speed_download} bytes/sec\n" https://www.google.com

# If slow, consider using a machine with better internet
```

### **Advanced Speed Techniques for Full Training**

#### **1. Parallel Training Sessions**
```bash
# Run multiple training sessions simultaneously on different machines
# Machine 1:
python3 search.py --n_generation 20 --valid_size 50 --max_workers 16 --expr_name "full_train_1"

# Machine 2:
python3 search.py --n_generation 20 --valid_size 50 --max_workers 16 --expr_name "full_train_2"
```

#### **2. Cloud Optimization**
```bash
# Use cloud instances with high CPU/memory
# AWS: c5.4xlarge (16 vCPUs, 32GB RAM)
# Google Cloud: n2-standard-16 (16 vCPUs, 64GB RAM)
# Azure: Standard_D16s_v3 (16 vCPUs, 64GB RAM)
```

#### **3. API Rate Limit Optimization**
```bash
# Monitor API usage to avoid rate limits
# Gemini typically allows 60 requests/minute
# With 16 workers, you're using ~1 request every 4 seconds per worker
# This should be well within limits
```

#### **4. Process Priority**
```bash
# Run with high priority (Linux)
nice -n -10 python3 search.py --n_generation 20 --valid_size 50 --max_workers 16

# Or use ionice for I/O priority
ionice -c 1 -n 0 python3 search.py --n_generation 20 --valid_size 50 --max_workers 16
```

### **Full Training Speed Benchmarks**

#### **Expected Times by Hardware**

| Hardware Configuration | Time Estimate | Cost |
|------------------------|---------------|------|
| **Basic Laptop** (4 cores, 8GB RAM) | 12-24 hours | Free |
| **Gaming PC** (8 cores, 16GB RAM) | 6-12 hours | Free |
| **Workstation** (16 cores, 32GB RAM) | 3-6 hours | Free |
| **Cloud Instance** (32 cores, 64GB RAM) | 1-3 hours | $2-5/hour |

#### **Speed Optimization Checklist**

- [ ] **CPU**: Using all available cores (`--max_workers` = cores - 2)
- [ ] **Memory**: At least 16GB RAM available
- [ ] **Storage**: SSD with 10GB+ free space
- [ ] **Network**: Fast, stable internet connection
- [ ] **API**: Google API key configured and working
- [ ] **Process**: Running with high priority
- [ ] **Background**: No other heavy processes running

### **Monitoring Full Training Performance**

#### **1. Real-time Monitoring**
```bash
# Monitor CPU usage
htop  # or top

# Monitor memory usage
free -h

# Monitor disk I/O
iotop  # or iostat

# Monitor network
iftop  # or nethogs
```

#### **2. Progress Tracking**
```bash
# Check training progress
tail -f results/full_speed_optimized_archive.json

# Monitor API calls
grep "API" /var/log/syslog  # or check your API dashboard
```

#### **3. Performance Alerts**
```bash
# Set up monitoring for:
# - CPU usage > 90%
# - Memory usage > 90%
# - Disk space < 5GB
# - Network errors
```

### **Troubleshooting Full Training Speed Issues**

#### **1. If Training is Too Slow**
```bash
# Check what's bottlenecking
# CPU: htop shows high usage
# Memory: free -h shows low available
# Network: ping google.com shows high latency
# Disk: iostat shows high I/O wait
```

#### **2. If API Calls are Failing**
```bash
# Reduce workers to avoid rate limits
python3 search.py --max_workers 8  # instead of 16

# Check API quota
# Go to Google AI Studio dashboard
```

#### **3. If Memory is Running Out**
```bash
# Reduce workers
python3 search.py --max_workers 8

# Or use a machine with more RAM
```

### **Optimal Full Training Configuration**

#### **For Most Users** (Balanced Speed/Reliability)
```bash
python3 search.py \
  --n_generation 20 \
  --valid_size 50 \
  --debug_max 2 \
  --max_workers 16 \
  --expr_name "full_balanced"
```

#### **For High-End Machines** (Maximum Speed)
```bash
python3 search.py \
  --n_generation 20 \
  --valid_size 50 \
  --debug_max 1 \
  --max_workers 32 \
  --expr_name "full_max_speed"
```

#### **For Cloud/Server** (Enterprise)
```bash
python3 search.py \
  --n_generation 20 \
  --valid_size 50 \
  --debug_max 2 \
  --max_workers 64 \
  --expr_name "full_enterprise"
```

### **Expected Full Training Timeline**

```
Hour 0-1:   Initial agent evaluation (7 agents × 50 problems)
Hour 1-2:   Generation 1 (first new agent)
Hour 2-3:   Generation 2 (second new agent)
...
Hour 18-20: Generation 20 (final new agent)
Total: ~20 hours (depending on hardware)
```

**Key**: The first hour is the slowest because it evaluates all 7 initial agents. Subsequent generations are faster because they only evaluate 1 new agent each.
