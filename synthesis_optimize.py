"""
Synthesis Prompt Optimizer

Uses Opik Optimizer to automatically improve agent prompts based on evaluation metrics.
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()

# Import optimizer - may need adjustment based on actual package API
try:
    from opik_optimizer import MetaPromptOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    console.print("[yellow]Warning: opik-optimizer not installed. Run: pip install opik-optimizer[/yellow]")


# ============================================================================
# Current Prompts from synthesis_app.py
# ============================================================================

PROMPTS = {
    "plan_agent1": {
        "name": "PlanAgent1 - Role Generation",
        "current": """You are a job market analysis agent. Based on the user's passion:
1. Analyze the job market
2. Generate EXACTLY 3 specific job roles

OUTPUT FORMAT (REQUIRED):
Return your response with a clear section titled "TOP 3 ROLES:" followed by exactly 3 numbered roles.

Example:
TOP 3 ROLES:
1. [Role Name 1] - Brief description
2. [Role Name 2] - Brief description
3. [Role Name 3] - Brief description

Also include market analysis before the roles list.

STYLE: Be direct and user-friendly. Present information clearly without explaining your reasoning process."""
    },
    
    "plan_agent2": {
        "name": "PlanAgent2 - Learning Path",
        "current": """You are a curriculum specialist. Based on the selected role:
1. Generate a structured learning path
2. List 5-7 key topics to master

OUTPUT FORMAT (REQUIRED):
Include a section titled "KEY TOPICS:" with numbered topics.

Example:
KEY TOPICS:
1. Topic 1
2. Topic 2
3. Topic 3
... (up to 7 topics)

Include brief descriptions for each topic.

STYLE: Be direct and user-friendly. Start with the content immediately - do NOT say things like "Okay, here's..." or "Based on the returned result...". Just present the learning path clearly."""
    },
    
    "do_agent": {
        "name": "DoAgent - Test Generation",
        "current": """You are a test specialist. Generate a multiple choice test.

CRITICAL: The test questions MUST be relevant to the ROLE and TOPIC specified.
For example:
- If the role is "Freelance Illustrator" and topic is "Fundamentals", ask about illustration fundamentals (color theory, composition, drawing basics).
- If the role is "Data Scientist" and topic is "Fundamentals", ask about data science fundamentals (statistics, pandas, data cleaning).

OUTPUT FORMAT (REQUIRED):
Generate exactly 5 questions in this format:

Q1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: [A/B/C/D]

Q2: [Question text]
... and so on for all 5 questions.

Make questions challenging but fair, and ALWAYS relevant to the specific role and topic."""
    },
    
    "go_agent": {
        "name": "GoAgent - Community/Projects",
        "current": """You are a career development specialist. Based on the user's role and completed topics:

For COMMUNITY requests:
- List relevant online communities (Discord, Slack, Reddit) with direct links if possible
- Suggest professional networks and meetups
- Recommend conferences and events
- Include specific subreddit names, Discord server names, or Slack workspace names

For PROJECT requests:
- Generate a detailed project plan in markdown format
- Include project goals, milestones, and deliverables
- Suggest technologies and resources needed

IMPORTANT: Provide complete, actionable information. Do NOT ask follow-up questions.
Do NOT end your response with questions like "What area are you focused on?" or "What is your experience level?"
Give comprehensive recommendations based on the information provided."""
    }
}


# ============================================================================
# Scoring Metrics for Optimization
# ============================================================================

def score_learning_path(output: str) -> float:
    """Score a learning path output."""
    score = 0.0
    
    # Check for numbered list
    if any(f"{i}." in output for i in range(1, 8)):
        score += 0.3
    
    # Check for KEY TOPICS section
    if "KEY TOPICS:" in output.upper() or "TOPICS:" in output.upper():
        score += 0.3
    
    # Check for fundamentals mention
    if "fundamental" in output.lower() or "basic" in output.lower():
        score += 0.2
    
    # Check for descriptions (sentences after topic names)
    lines = output.split('\n')
    described_topics = sum(1 for line in lines if '. ' in line and len(line) > 50)
    if described_topics >= 3:
        score += 0.2
    
    return score


def score_role_relevance(output: str) -> float:
    """Score role generation output."""
    score = 0.0
    
    # Check for TOP 3 ROLES section
    if "TOP 3 ROLES:" in output.upper() or "ROLES:" in output.upper():
        score += 0.3
    
    # Check for numbered list
    if all(f"{i}." in output for i in [1, 2, 3]):
        score += 0.3
    
    # Check for descriptions (dashes or colons after role names)
    if output.count(" - ") >= 2 or output.count(": ") >= 2:
        score += 0.2
    
    # Check for market analysis
    if "market" in output.lower() or "demand" in output.lower() or "opportunity" in output.lower():
        score += 0.2
    
    return score


def score_test_questions(output: str) -> float:
    """Score test question generation output."""
    score = 0.0
    
    # Check for question format
    q_count = sum(1 for i in range(1, 6) if f"Q{i}:" in output or f"Q{i}." in output)
    score += min(0.4, q_count * 0.08)
    
    # Check for options A-D
    if all(f"{opt})" in output for opt in ['A', 'B', 'C', 'D']):
        score += 0.3
    
    # Check for correct answers
    if "Correct:" in output or "correct:" in output:
        score += 0.3
    
    return score


def score_output_style(output: str) -> float:
    """Score output style (penalize preambles)."""
    red_flags = [
        "okay, here's",
        "based on the returned result",
        "i will",
        "let me",
        "here is what i found"
    ]
    
    output_lower = output.lower()
    issues = sum(1 for flag in red_flags if flag in output_lower)
    
    return max(0, 1.0 - (issues * 0.25))


# ============================================================================
# Optimizer Runner
# ============================================================================

def run_optimization(prompt_key: str = "plan_agent2"):
    """Run prompt optimization for a specific agent."""
    
    if not OPTIMIZER_AVAILABLE:
        console.print("[red]Cannot run optimization: opik-optimizer not installed[/red]")
        console.print("[dim]Install with: pip install opik-optimizer[/dim]")
        return None
    
    if prompt_key not in PROMPTS:
        console.print(f"[red]Unknown prompt key: {prompt_key}[/red]")
        console.print(f"[dim]Available: {list(PROMPTS.keys())}[/dim]")
        return None
    
    prompt_info = PROMPTS[prompt_key]
    console.print(Panel(
        f"Optimizing: [bold]{prompt_info['name']}[/bold]",
        title="🔧 Opik Optimizer",
        border_style="cyan"
    ))
    
    # Select appropriate scoring function
    scoring_funcs = {
        "plan_agent1": score_role_relevance,
        "plan_agent2": score_learning_path,
        "do_agent": score_test_questions,
        "go_agent": score_output_style,
    }
    
    scoring_func = scoring_funcs.get(prompt_key, score_output_style)
    
    console.print(f"\n[dim]Initial prompt:[/dim]")
    console.print(f"[cyan]{prompt_info['current'][:200]}...[/cyan]")
    
    try:
        import opik
        from typing import Any
        from opik_optimizer import ChatPrompt, MetaPromptOptimizer
        from opik.evaluation.metrics.score_result import ScoreResult
        
        # Get or create the evaluation dataset
        client = opik.Opik()
        dataset = client.get_or_create_dataset("synthesis-optimization-dataset")
        
        # Add sample data if needed
        sample_data = [
            {"input": "I love creating art and visual designs"},
            {"input": "I'm passionate about data analysis and patterns"},
            {"input": "I enjoy building software and solving problems"},
        ]
        dataset.insert(sample_data)
        
        # Create metric function matching opik-optimizer API
        # Must take dataset_item and llm_output, return ScoreResult
        def optimization_metric(dataset_item: dict[str, Any], llm_output: str) -> ScoreResult:
            score_value = scoring_func(llm_output)
            return ScoreResult(
                value=score_value,
                name=prompt_key + "_score",
                reason=f"Structural score based on output format: {score_value:.2f}"
            )
        
        # Create ChatPrompt object
        prompt = ChatPrompt(
            messages=[
                {"role": "system", "content": prompt_info['current']},
                {"role": "user", "content": "{input}"}
            ]
        )
        # Ensure OpenRouter API key is available for LiteLLM
        # Some internal calls may look for OPENAI_API_KEY, so we set it via OpenRouter
        import os
        if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        
        # Initialize optimizer with free OpenRouter model
        optimizer = MetaPromptOptimizer(
            model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
            verbose=1
        )
        
        # Run optimization with required parameters
        console.print("\n[bold green]Running optimization...[/bold green]")
        result = optimizer.optimize_prompt(
            prompt=prompt,
            dataset=dataset,
            metric=optimization_metric,
            max_trials=5,
            project_name="synthesis"
        )

        
        console.print("\n[bold green]✓ Optimization complete![/bold green]")
        
        # Display results using built-in method
        result.display()
        
        # Handle result - structure may vary by version
        best_prompt = getattr(result, 'best_prompt', None) or getattr(result, 'prompt', None) or str(result)
        if hasattr(best_prompt, 'messages'):
            best_prompt = best_prompt.messages[0].get('content', str(best_prompt))
        console.print(Panel(str(best_prompt)[:500] + "..." if len(str(best_prompt)) > 500 else str(best_prompt), title="Best Prompt", border_style="green"))
        
        # Save optimized prompt
        output_file = f"optimized_{prompt_key}.txt"
        with open(output_file, 'w') as f:
            f.write(f"# Optimized prompt for {prompt_info['name']}\n\n")
            f.write(str(best_prompt))
        console.print(f"[green]✓ Saved to {output_file}[/green]")

        
        return result
        
    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")

        console.print("[dim]This may require additional setup or a different model.[/dim]")
        return None


def list_prompts():
    """List all available prompts that can be optimized."""
    console.print("\n[bold]Available prompts for optimization:[/bold]\n")
    for key, info in PROMPTS.items():
        console.print(f"  [cyan]{key}[/cyan] - {info['name']}")
    console.print("\n[dim]Usage: python synthesis_optimize.py <prompt_key>[/dim]")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--list":
            list_prompts()
        else:
            run_optimization(arg)
    else:
        # Default: optimize the learning path prompt (lowest scoring)
        console.print("[bold]Synthesis Prompt Optimizer[/bold]\n")
        console.print("[dim]Use --list to see available prompts[/dim]")
        console.print("[dim]Use <prompt_key> to optimize a specific prompt[/dim]\n")
        
        # Show current prompt scores
        console.print("[bold]Quick check - scoring sample outputs:[/bold]")
        
        sample_learning_path = """KEY TOPICS:
1. Fundamentals - Basic concepts
2. Core Skills - Essential techniques
3. Advanced Topics - Specialized knowledge"""
        
        console.print(f"  Learning Path Score: {score_learning_path(sample_learning_path):.2f}")
        console.print(f"  Style Score (clean): {score_output_style('Here are the topics...'):.2f}")
        verbose_sample = "Okay, here's what I found..."
        console.print(f"  Style Score (verbose): {score_output_style(verbose_sample):.2f}")

        
        console.print("\n[yellow]Run with a prompt key to start optimization[/yellow]")
        list_prompts()
