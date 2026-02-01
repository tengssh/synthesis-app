"""
Synthesis Evaluation Script

Runs evaluations on the Synthesis app agents using Opik's evaluate framework.
Dynamically loads and uses metrics defined in opik_evals.yaml.
"""

import os
import re
import yaml
import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import base_metric, score_result
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

console = Console()

# ============================================================================
# YAML Config Loader
# ============================================================================

def load_eval_config(yaml_path: str = "opik_evals.yaml") -> dict:
    """Load evaluation configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# Dynamic Metric Factory - Creates metrics from YAML definitions
# ============================================================================

class YAMLDefinedMetric(base_metric.BaseMetric):
    """
    A metric dynamically created from YAML configuration.
    Supports both LLM judge and heuristic evaluation types.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", "unnamed_metric")
        self.description = config.get("description", "")
        self.eval_type = config.get("type", "heuristic")
        self.prompt_template = config.get("prompt", "")
        self.rules = config.get("rules", [])
        self.scoring = config.get("scoring", {})
    
    def score(self, input: str = "", output: str = "", **kwargs) -> score_result.ScoreResult:
        """Score based on the evaluation type defined in YAML."""
        
        if self.eval_type == "llm_judge":
            return self._score_llm_judge(input, output, **kwargs)
        elif self.eval_type == "heuristic":
            return self._score_heuristic(input, output, **kwargs)
        else:
            return score_result.ScoreResult(
                name=self.name,
                value=0.0,
                reason=f"Unknown evaluation type: {self.eval_type}"
            )
    
    def _score_llm_judge(self, input: str, output: str, **kwargs) -> score_result.ScoreResult:
        """
        Score using an LLM judge.
        In production, this would call the LLM with the prompt template.
        For now, we use heuristics based on the prompt's expected output format.
        """
        # Parse expected format from prompt
        prompt = self.prompt_template.lower()
        
        # Check for score patterns mentioned in the prompt
        if "score:" in prompt or "1-10" in prompt:
            # This is a numeric scoring prompt - use heuristics
            return self._score_by_content_analysis(input, output)
        elif "complexity level:" in prompt:
            # Categorical scoring
            return self._score_categorical(output)
        else:
            return self._score_by_content_analysis(input, output)
    
    def _score_by_content_analysis(self, input: str, output: str) -> score_result.ScoreResult:
        """Analyze content for quality indicators."""
        score_value = 0.5  # Start neutral
        reasons = []
        
        # Check for structured output
        if any(f"{i}." in output for i in range(1, 10)):
            score_value += 0.15
            reasons.append("has_numbered_list")
        
        # Check for section headers
        if ":" in output and any(word.isupper() for word in output.split()[:20]):
            score_value += 0.1
            reasons.append("has_sections")
        
        # Check output length (meaningful content)
        if len(output) > 200:
            score_value += 0.1
            reasons.append("substantial_content")
        
        # Check for actionable content (links, specific names)
        if "http" in output or "@" in output or "/" in output.lower():
            score_value += 0.1
            reasons.append("has_actionable_links")
        
        # Penalize verbose preambles
        preambles = ["okay, here's", "let me", "i will", "based on"]
        if any(p in output.lower()[:100] for p in preambles):
            score_value -= 0.15
            reasons.append("verbose_preamble")
        
        score_value = max(0.0, min(1.0, score_value))
        
        return score_result.ScoreResult(
            name=self.name,
            value=score_value,
            reason=", ".join(reasons) if reasons else "default_score"
        )
    
    def _score_categorical(self, output: str) -> score_result.ScoreResult:
        """Score categorical outputs."""
        categories = self.scoring.get("categories", [])
        output_lower = output.lower()
        
        for cat in categories:
            if cat.lower() in output_lower:
                return score_result.ScoreResult(
                    name=self.name,
                    value=1.0,
                    reason=f"Matched category: {cat}"
                )
        
        return score_result.ScoreResult(
            name=self.name,
            value=0.5,
            reason="No category match found"
        )
    
    def _score_heuristic(self, input: str, output: str, **kwargs) -> score_result.ScoreResult:
        """Score using heuristic rules from YAML."""
        total_weight = sum(rule.get("weight", 0) for rule in self.rules)
        if total_weight == 0:
            total_weight = 100
        
        earned_weight = 0
        matched_rules = []
        
        for rule in self.rules:
            check = rule.get("check", "")
            weight = rule.get("weight", 0)
            
            # Evaluate each check type
            if check == "contains_ingredients" or check == "contains_goals":
                if "goal" in output.lower() or "objective" in output.lower():
                    earned_weight += weight
                    matched_rules.append(check)
            
            elif check == "contains_instructions" or check == "contains_milestones":
                if "step" in output.lower() or "milestone" in output.lower() or any(f"{i}." in output for i in range(1, 10)):
                    earned_weight += weight
                    matched_rules.append(check)
            
            elif check == "contains_timing":
                if "time" in output.lower() or "week" in output.lower() or "day" in output.lower():
                    earned_weight += weight
                    matched_rules.append(check)
            
            elif check == "contains_servings" or check == "contains_deliverables":
                if "deliverable" in output.lower() or "output" in output.lower():
                    earned_weight += weight
                    matched_rules.append(check)
            
            elif check == "contains_tips" or check == "contains_resources":
                if "tip" in output.lower() or "resource" in output.lower() or "learn" in output.lower():
                    earned_weight += weight
                    matched_rules.append(check)
            
            elif check == "contains_technologies":
                if "tool" in output.lower() or "technology" in output.lower() or "software" in output.lower():
                    earned_weight += weight
                    matched_rules.append(check)
            
            elif check == "contains_topics":
                if "topic" in output.lower() or "KEY TOPICS:" in output.upper():
                    earned_weight += weight
                    matched_rules.append(check)
            
            elif check == "has_progression":
                if "fundamental" in output.lower() or "basic" in output.lower() or "advanced" in output.lower():
                    earned_weight += weight
                    matched_rules.append(check)
            
            elif check == "includes_resources":
                if "resource" in output.lower() or "http" in output.lower():
                    earned_weight += weight
                    matched_rules.append(check)
        
        score_value = earned_weight / total_weight
        
        return score_result.ScoreResult(
            name=self.name,
            value=score_value,
            reason=f"Matched: {matched_rules}" if matched_rules else "No rules matched"
        )


def create_metrics_from_yaml(config: dict) -> list:
    """Create metric instances from YAML configuration."""
    metrics = []
    
    for eval_config in config.get("evaluations", []):
        metric = YAMLDefinedMetric(eval_config)
        metrics.append(metric)
        console.print(f"  [dim]Loaded metric: {metric.name} ({metric.eval_type})[/dim]")
    
    return metrics


# ============================================================================
# Evaluation Dataset
# ============================================================================

def create_evaluation_dataset():
    """Create or get the evaluation dataset for Synthesis app."""
    client = opik.Opik()
    
    dataset = client.get_or_create_dataset("synthesis-eval-dataset")
    
    # Add test cases if dataset is empty
    test_cases = [
        {
            "input": "I love drawing and creating visual art",
            "context": "role_generation",
            "expected_keywords": ["illustrator", "designer", "artist"]
        },
        {
            "input": "I'm passionate about data and finding patterns",
            "context": "role_generation", 
            "expected_keywords": ["data scientist", "analyst", "machine learning"]
        },
        {
            "input": "Generate test for role: UI/UX Designer, topic: Fundamentals",
            "context": "test_generation",
            "expected_keywords": ["design", "user", "interface", "wireframe"]
        },
        {
            "input": "Generate test for role: Data Scientist, topic: Advanced Topics",
            "context": "test_generation",
            "expected_keywords": ["model", "algorithm", "data", "analysis"]
        },
    ]
    
    # Insert test cases
    dataset.insert(test_cases)
    
    return dataset


# ============================================================================
# Evaluation Task
# ============================================================================

def evaluation_task(item: dict) -> dict:
    """
    The evaluation task that maps dataset items to outputs.
    
    In a full implementation, this would:
    1. Run the actual agents with the input
    2. Return the agent output for scoring
    
    For demo purposes, we return mock outputs.
    """
    # Mock output - in production, run actual agents here
    mock_outputs = {
        "role_generation": """
TOP 3 ROLES:
1. Graphic Designer - Creating visual content for brands and marketing materials
2. Illustrator - Drawing for books, games, and digital media
3. UI/UX Designer - Designing user interfaces and experiences

Market Analysis: The creative industry shows strong demand for visual designers,
with opportunities in tech, advertising, and entertainment sectors.
""",
        "test_generation": """
Q1: What is the primary purpose of user research in UX design?
A) To make the product look pretty
B) To understand user needs and behaviors
C) To increase development speed
D) To reduce testing costs
Correct: B

Q2: Which tool is commonly used for creating wireframes?
A) Photoshop
B) Figma
C) Excel
D) PowerPoint
Correct: B
"""
    }
    
    context = item.get("context", "role_generation")
    return {
        "output": mock_outputs.get(context, "No output"),
        "input": item.get("input", "")
    }


# ============================================================================
# Main Evaluation Runner
# ============================================================================

def run_evaluation(use_yaml_metrics: bool = True):
    """Run the full evaluation suite."""
    console.print("[bold cyan]🔍 Starting Synthesis App Evaluation...[/bold cyan]")
    
    # Load YAML config
    config = load_eval_config()
    console.print(f"[green]✓ Loaded {len(config.get('evaluations', []))} evaluation definitions from opik_evals.yaml[/green]")
    
    # Create dataset
    dataset = create_evaluation_dataset()
    console.print(f"[green]✓ Using dataset: {dataset.name}[/green]")
    
    # Create metrics from YAML
    if use_yaml_metrics:
        console.print("\n[bold]Creating metrics from YAML:[/bold]")
        metrics = create_metrics_from_yaml(config)
    else:
        # Legacy mode: just use first 2 metrics from YAML as fallback
        console.print("[yellow]Legacy mode: using subset of YAML metrics[/yellow]")
        all_metrics = create_metrics_from_yaml(config)
        metrics = all_metrics[:2] if len(all_metrics) >= 2 else all_metrics
    
    console.print(f"\n[bold]Running evaluation with {len(metrics)} metrics...[/bold]")
    
    # Run evaluation
    result = evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=metrics,
        experiment_name="synthesis-eval-yaml",
        project_name="synthesis"
    )
    
    # Print results
    console.print("\n[bold cyan]📈 Evaluation Results:[/bold cyan]")
    console.print("=" * 50)
    
    scores = result.aggregate_evaluation_scores()
    for metric_name, statistics in scores.aggregated_scores.items():
        avg = statistics.mean if hasattr(statistics, 'mean') else statistics
        console.print(f"  [bold]{metric_name}[/bold]: {avg:.2f}" if isinstance(avg, float) else f"  [bold]{metric_name}[/bold]: {statistics}")
    
    console.print("\n[green]✅ Evaluation complete! Check Opik dashboard for details.[/green]")
    return result


if __name__ == "__main__":
    import sys
    
    use_yaml = "--legacy" not in sys.argv
    
    if "--help" in sys.argv:
        console.print("[bold]Synthesis Evaluation Script[/bold]")
        console.print("\nUsage: python synthesis_eval.py [options]")
        console.print("\nOptions:")
        console.print("  --legacy    Use hardcoded metrics instead of YAML")
        console.print("  --help      Show this help message")
    else:
        run_evaluation(use_yaml_metrics=use_yaml)
