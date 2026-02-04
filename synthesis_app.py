"""
Synthesis - Knowledge OS with Google ADK and Opik Integration
=======================================================

A multi-agent system for knowledge synthesis and research using Google's Agent Development Kit
with Comet Opik observability integration.

Interactive Multi-Step Workflow:
1. Enter passion → Get top 3 roles
2. Select role → Get learning path
3. Select topic → Take test
4. Complete tests → Earn badge
5. Save results → Find community or generate project plan

Agents:
- PlanAgent1: Based on user's passion, perform analysis on current market and future trend, then generate the top 3 roles.
- PlanAgent2: Based on user's selected role, generate a learning path using mindmap visualization.
- DoAgent: Based on the selected topic, generate multiple choice tests.
- GoAgent: Generate badges and project plans.
"""

import os
import asyncio
import warnings
import time
import json
import re
from functools import wraps
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

# --- Opik Monkey-patch start ---
try:
    from opik.llm_usage import google_usage
    from pydantic import Field
    from pydantic.fields import FieldInfo
    
    # Make candidates_token_count optional to handle cases where Gemini doesn't return it
    if hasattr(google_usage, "GoogleGeminiUsage"):
        google_usage.GoogleGeminiUsage.model_fields['candidates_token_count'] = FieldInfo(
            annotation=Optional[int], 
            default=None,
            description="Number of tokens in the response(s)."
        )
        google_usage.GoogleGeminiUsage.model_rebuild(force=True)
except Exception:
    # Fail silently or log if opik is not installed or structure is different
    pass
# --- Opik Monkey-patch end ---

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from opik.integrations.adk import OpikTracer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")

# Remove GEMINI_API_KEY if both are set to avoid warnings
if os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ.pop("GEMINI_API_KEY", None)


# Initialize Opik tracer with configuration
opik_tracer = OpikTracer(
    name="synthesis-agent-system",
    tags=["synthesis", "multi-agent"],
    metadata={
        "environment": "development",
        "version": "2.0.0"
    },
    project_name="synthesis"
)


# ============================================================================
# Data Classes for State Management
# ============================================================================

@dataclass
class TestResult:
    """Stores result of a single test."""
    topic: str
    score: int
    total: int
    passed: bool
    answers: List[str] = field(default_factory=list)


@dataclass
class UserProgress:
    """Tracks user's progress through the entire workflow."""
    passion: str = ""
    available_roles: List[str] = field(default_factory=list)
    selected_role: str = ""
    learning_path: List[str] = field(default_factory=list)
    completed_topics: List[str] = field(default_factory=list)
    test_results: List[dict] = field(default_factory=list)
    badge_earned: bool = False
    badge_level: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def save_to_json(self, filename: str = "synthesis_results.json"):
        """Save progress to JSON file."""
        with open(filename, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        console.print(f"[green]✓ Results saved to {filename}[/green]")
    
    def calculate_average_score(self) -> float:
        """Calculate average test score."""
        if not self.test_results:
            return 0.0
        total_score = sum(r['score'] for r in self.test_results)
        total_possible = sum(r['total'] for r in self.test_results)
        return (total_score / total_possible * 100) if total_possible > 0 else 0.0
    
    @classmethod
    def load_from_json(cls, filename: str = "synthesis_results.json") -> 'UserProgress':
        """Load progress from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            progress = cls(
                passion=data.get('passion', ''),
                available_roles=data.get('available_roles', []),
                selected_role=data.get('selected_role', ''),
                learning_path=data.get('learning_path', []),
                completed_topics=data.get('completed_topics', []),
                test_results=data.get('test_results', []),
                badge_earned=data.get('badge_earned', False),
                badge_level=data.get('badge_level', ''),
                timestamp=data.get('timestamp', datetime.now().isoformat())
            )
            return progress
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            console.print("[red]Error: Invalid JSON file.[/red]")
            return None


# ============================================================================
# Retry Decorator for API Overload
# ============================================================================

def retry_on_overload(max_retries=5, initial_delay=2, backoff_factor=2):
    """Decorator to retry function calls with exponential backoff on 503 errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    last_exception = e
                    
                    if "503" in error_str and ("overloaded" in error_str.lower() or "unavailable" in error_str.lower()):
                        if attempt < max_retries - 1:
                            console.print(f"[yellow]⚠️  API overloaded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})[/yellow]")
                            time.sleep(delay)
                            delay *= backoff_factor
                            continue
                    raise
            
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# UI Helper Functions
# ============================================================================

def display_header():
    """Display application header."""
    console.print(Panel.fit(
        "[bold cyan]🧠 Synthesis App[/bold cyan]\n"
        "[dim]Knowledge OS - Learn, Test, Achieve[/dim]\n"
        "Powered by Google ADK with Opik Observability",
        border_style="cyan"
    ))
    console.print()


def display_menu(title: str, options: List[str], allow_exit: bool = True) -> int:
    """Display numbered menu and get user selection.
    
    Returns:
        Index of selected option (0-based), or -1 if exit selected
    """
    console.print(f"\n[bold cyan]━━━ {title} ━━━[/bold cyan]")
    for i, option in enumerate(options, 1):
        console.print(f"  [bold]{i}.[/bold] {option}")
    if allow_exit:
        console.print(f"  [bold]0.[/bold] [dim]Exit/Back[/dim]")
    
    while True:
        choice = console.input("\n[yellow]Enter your choice:[/yellow] ").strip()
        if choice == "" and not allow_exit:
            return 0  # Default to first option
        if choice == "0" and allow_exit:
            return -1
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        console.print("[red]Invalid choice. Please try again.[/red]")


def display_badge(progress: UserProgress):
    """Display achievement badge using compact Rich Table in a Panel."""
    avg_score = progress.calculate_average_score()
    completed_count = len(progress.completed_topics)
    total_topics = len(progress.learning_path) if progress.learning_path else 1
    completion_ratio = completed_count / total_topics
    
    # Determine badge level based on BOTH score AND topic completion
    if avg_score >= 90 and completion_ratio >= 0.8:
        badge_level = "🏆 GOLD"
        badge_color = "yellow"
    elif avg_score >= 70 and completion_ratio >= 0.5:
        badge_level = "🥈 SILVER"
        badge_color = "white"
    else:
        badge_level = "🥉 BRONZE"
        badge_color = "orange3"
    
    progress.badge_level = badge_level
    progress.badge_earned = True
    
    role_display = progress.selected_role[:35] if len(progress.selected_role) > 35 else progress.selected_role
    
    # Compact table inside panel
    from rich.box import SIMPLE
    table = Table(show_header=False, box=SIMPLE, padding=(0, 1), expand=False)
    table.add_column("Label", style="dim", width=18)
    table.add_column("Value", style=f"bold {badge_color}")
    
    table.add_row("Badge Level", badge_level)
    table.add_row("Role", role_display)
    table.add_row("Topics Completed", f"{completed_count}/{total_topics}")
    table.add_row("Average Score", f"{avg_score:.1f}%")
    table.add_row("Date", progress.timestamp[:10])
    
    console.print(Panel(table, title="🎓 ACHIEVEMENT UNLOCKED", border_style=badge_color, expand=False))



def display_progress_table(progress: UserProgress):
    """Display progress summary as a table."""
    table = Table(title="📊 Your Progress Summary", show_header=True, header_style="bold cyan")
    table.add_column("Topic", style="dim")
    table.add_column("Type", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Status", justify="center")
    
    for result in progress.test_results:
        status = "✅ Pass" if result['passed'] else "❌ Fail"
        score_str = f"{result['score']}/{result['total']}"
        test_type = "🔄 Challenge" if result.get('is_challenge', False) else "📝 Test"
        table.add_row(result['topic'], test_type, score_str, status)
    
    console.print(table)



# ============================================================================
# Agent Tools
# ============================================================================

def search_job_market(passion: str) -> str:
    """Search a job market for detailed information."""
    return f"Current job market information for '{passion}': include market size, growth rate, and top companies."


def get_top_3_roles(market_info: str) -> str:
    """Get top 3 roles for the given market information."""
    return f"Top 3 roles for '{market_info}': include job description, salary, and future development."


def search_learning_path(role: str) -> str:
    """Search a learning path for the given role."""
    return f"Learning path for '{role}': include hierarchical structure and significance."


def generate_test_questions(topic: str) -> str:
    """Generate test questions for a topic."""
    return f"Generate 5 multiple choice questions for the topic '{topic}' with 4 options each and indicate the correct answer."


def find_community_resources(role: str, skills: list) -> str:
    """Find community resources for networking."""
    skills_str = ", ".join(skills) if skills else "general"
    return f"Find online communities, forums, and networking opportunities for '{role}' with skills in {skills_str}."


def generate_project_ideas(role: str, skills: list) -> str:
    """Generate project ideas based on role and skills."""
    skills_str = ", ".join(skills) if skills else "general"
    return f"Generate practical project ideas for '{role}' using skills: {skills_str}. Include project scope, technologies, and expected outcomes."


# ============================================================================
# Agent Definitions
# ============================================================================

def create_agents(model: str = "gemma-3-27b-it"):
    """Create and return all agents with specified model."""
    
    # PlanAgent1: Generate top 3 roles
    plan_agent1 = Agent(
        name="PlanAgent1",
        model=model,
        instruction="""You are a job market analysis agent. Based on the user's passion:
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
        
        STYLE: Be direct and user-friendly. Present information clearly without explaining your reasoning process.""",
        output_key="market_analysis",
        tools=[search_job_market, get_top_3_roles],
        before_agent_callback=opik_tracer.before_agent_callback,
        after_agent_callback=opik_tracer.after_agent_callback,
        before_model_callback=opik_tracer.before_model_callback,
        after_model_callback=opik_tracer.after_model_callback,
        before_tool_callback=opik_tracer.before_tool_callback,
        after_tool_callback=opik_tracer.after_tool_callback,
    )
    
    # PlanAgent2: Generate learning path (OPTIMIZED via opik-optimizer)
    plan_agent2 = Agent(
        name="PlanAgent2",
        model=model,
        instruction="""You are an expert learning path designer with deep expertise in career development and skill progression. Your mission is to create personalized learning roadmaps that are practical and actionable.

FOR EACH LEARNING PATH:
1. Analyze the interest area carefully
2. Identify foundational to advanced progression
3. Focus on industry-relevant skills
4. Ensure logical skill dependencies

OUTPUT STRUCTURE:
KEY TOPICS:
[5-7 numbered topics with brief descriptions]

RULES:
- Each topic must build on previous ones
- Include both theoretical and practical elements
- Focus on current industry standards
- Descriptions should be clear and actionable""",
        output_key="learning_path",
        tools=[search_learning_path],
        before_agent_callback=opik_tracer.before_agent_callback,
        after_agent_callback=opik_tracer.after_agent_callback,
        before_model_callback=opik_tracer.before_model_callback,
        after_model_callback=opik_tracer.after_model_callback,
        before_tool_callback=opik_tracer.before_tool_callback,
        after_tool_callback=opik_tracer.after_tool_callback,
    )
    
    # DoAgent: Generate tests
    do_agent = Agent(
        name="DoAgent",
        model=model,
        instruction="""You are a test specialist. Generate a multiple choice test.
        
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
        
        Make questions challenging but fair, and ALWAYS relevant to the specific role and topic.""",
        output_key="test_questions",
        tools=[generate_test_questions],
        before_agent_callback=opik_tracer.before_agent_callback,
        after_agent_callback=opik_tracer.after_agent_callback,
        before_model_callback=opik_tracer.before_model_callback,
        after_model_callback=opik_tracer.after_model_callback,
        before_tool_callback=opik_tracer.before_tool_callback,
        after_tool_callback=opik_tracer.after_tool_callback,
    )
    
    # GoAgent: Community and projects
    go_agent = Agent(
        name="GoAgent",
        model=model,
        instruction="""You are a career development specialist. Based on the user's role and completed topics:
        
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
        Give comprehensive recommendations based on the information provided.""",
        output_key="career_guidance",
        tools=[find_community_resources, generate_project_ideas],
        before_agent_callback=opik_tracer.before_agent_callback,
        after_agent_callback=opik_tracer.after_agent_callback,
        before_model_callback=opik_tracer.before_model_callback,
        after_model_callback=opik_tracer.after_model_callback,
        before_tool_callback=opik_tracer.before_tool_callback,
        after_tool_callback=opik_tracer.after_tool_callback,
    )
    
    return plan_agent1, plan_agent2, do_agent, go_agent


# ============================================================================
# Agent Runner Functions
# ============================================================================

@retry_on_overload(max_retries=5, initial_delay=3, backoff_factor=2)
def run_agent(agent: Agent, prompt: str, app_name: str = "synthesis-app") -> str:
    """Run a single agent and return the response text."""
    response_text = ""
    
    async def _run():
        nonlocal response_text
        runner = InMemoryRunner(agent=agent, app_name=app_name)
        session = await runner.session_service.create_session(app_name=app_name, user_id="user_1")
        content = types.Content(role='user', parts=[types.Part(text=prompt)])
        
        async for event in runner.run_async(
            user_id="user_1",
            session_id=session.id,
            new_message=content
        ):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text + "\n"
    
    asyncio.run(_run())
    return response_text


def extract_roles_from_response(response: str) -> List[str]:
    """Extract role names from agent response."""
    roles = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        # Match patterns like "1. Role Name" or "1) Role Name" or just numbered lines after "TOP 3 ROLES"
        if re.match(r'^[1-3][\.\)]\s+', line):
            # Remove the number prefix and extract the role
            role = re.sub(r'^[1-3][\.\)]\s+', '', line)
            # Clean up - take text before any dash or description
            if ' - ' in role:
                role = role.split(' - ')[0].strip()
            if role:
                roles.append(role)
    
    # Fallback: if no roles found, try to find any line with role-like patterns
    if not roles:
        role_keywords = ['Developer', 'Engineer', 'Scientist', 'Analyst', 'Designer', 'Manager', 'Specialist', 'Architect']
        for line in lines:
            for keyword in role_keywords:
                if keyword in line:
                    # Extract a reasonable role name
                    clean_line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line.strip())
                    if clean_line and len(clean_line) < 100:
                        roles.append(clean_line.split(' - ')[0].strip())
                        break
            if len(roles) >= 3:
                break
    
    return roles[:3] if roles else ["General Specialist", "Technical Lead", "Domain Expert"]


def extract_topics_from_response(response: str) -> List[str]:
    """Extract learning topics from agent response."""
    topics = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        # Match numbered topics
        if re.match(r'^[1-7][\.\)]\s+', line):
            topic = re.sub(r'^[1-7][\.\)]\s+', '', line)
            # Clean up
            if ' - ' in topic:
                topic = topic.split(' - ')[0].strip()
            if topic and len(topic) < 100:
                topics.append(topic)
    
    return topics[:7] if topics else ["Fundamentals", "Core Concepts", "Advanced Topics"]


def parse_test_and_run(test_response: str, topic: str) -> dict:
    """Parse test questions from response and run interactive test."""
    console.print(Panel(f"[bold]📝 Test: {topic}[/bold]", border_style="blue"))
    
    # Parse questions
    questions = []
    lines = test_response.split('\n')
    current_q = None
    options = {}
    correct = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Q') and ':' in line:
            if current_q and options and correct:
                questions.append({'question': current_q, 'options': options.copy(), 'correct': correct})
            current_q = line.split(':', 1)[1].strip() if ':' in line else line
            options = {}
            correct = None
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            key = line[0]
            value = line[2:].strip()
            options[key] = value
        elif line.lower().startswith('correct:'):
            correct = line.split(':')[1].strip().upper()
    
    # Add last question
    if current_q and options and correct:
        questions.append({'question': current_q, 'options': options.copy(), 'correct': correct})
    
    # If parsing failed, create sample questions
    if not questions:
        questions = [
            {'question': f'What is a key concept in {topic}?', 'options': {'A': 'Option A', 'B': 'Option B', 'C': 'Option C', 'D': 'Option D'}, 'correct': 'A'},
            {'question': f'How is {topic} applied in practice?', 'options': {'A': 'Method A', 'B': 'Method B', 'C': 'Method C', 'D': 'Method D'}, 'correct': 'B'},
            {'question': f'What is an advanced aspect of {topic}?', 'options': {'A': 'Aspect A', 'B': 'Aspect B', 'C': 'Aspect C', 'D': 'Aspect D'}, 'correct': 'C'},
        ]
    
    # Run test
    score = 0
    total = len(questions)
    user_answers = []
    
    for i, q in enumerate(questions, 1):
        console.print(f"\n[bold]Question {i}/{total}:[/bold] {q['question']}")
        for key, value in q['options'].items():
            console.print(f"  {key}) {value}")
        
        while True:
            answer = console.input("\n[yellow]Your answer (A/B/C/D):[/yellow] ").strip().upper()
            if answer in ['A', 'B', 'C', 'D']:
                break
            console.print("[red]Please enter A, B, C, or D[/red]")
        
        user_answers.append(answer)
        if answer == q['correct']:
            console.print("[green]✓ Correct![/green]")
            score += 1
        else:
            console.print(f"[red]✗ Wrong. Correct answer: {q['correct']}[/red]")
    
    passed = score >= total * 0.6
    console.print(f"\n[bold]Result:[/bold] {score}/{total} - {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return {
        'topic': topic,
        'score': score,
        'total': total,
        'passed': passed
    }


# ============================================================================
# Main Interactive Workflow
# ============================================================================

def main():
    """Main function running the interactive multi-step workflow."""
    display_header()
    
    # Model selection (optional)
    model_options = [
        "gemma-3-27b-it",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview"
    ]
    model_choice = display_menu("Select AI Model (Enter for default)", model_options, allow_exit=False)
    selected_model = model_options[model_choice] if model_choice >= 0 else "gemma-3-27b-it"
    console.print(f"[dim]Using model: {selected_model}[/dim]\n")
    
    # Create agents with selected model
    plan_agent1, plan_agent2, do_agent, go_agent = create_agents(selected_model)
    
    # Check for existing progress file
    progress = None
    resume_mode = False
    
    if os.path.exists("synthesis_results.json"):
        console.print("[dim]Previous progress file found.[/dim]")
        startup_choice = display_menu("How would you like to start?", [
            "🆕 Start fresh (new session)",
            "📂 Resume from previous progress"
        ], allow_exit=True)
        
        if startup_choice == -1:
            console.print("[yellow]Goodbye![/yellow]")
            return
        elif startup_choice == 0:
            # Start fresh - archive old files with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Archive existing JSON
            if os.path.exists("synthesis_results.json"):
                archived_name = f"synthesis_results_{timestamp}.json"
                os.rename("synthesis_results.json", archived_name)
                console.print(f"[dim]Previous results archived to: {archived_name}[/dim]")
            
            # Archive existing project plan if exists
            if os.path.exists("project_plan.md"):
                archived_plan = f"project_plan_{timestamp}.md"
                os.rename("project_plan.md", archived_plan)
                console.print(f"[dim]Previous project plan archived to: {archived_plan}[/dim]")
            
            progress = UserProgress()
        else:
            # Resume from previous progress
            progress = UserProgress.load_from_json()
            if progress and progress.passion and progress.selected_role and progress.learning_path:
                resume_mode = True
                console.print(f"[green]✓ Loaded progress for:[/green] {progress.passion}")
                console.print(f"[dim]  Role: {progress.selected_role}[/dim]")
                console.print(f"[dim]  Completed topics: {len(progress.completed_topics)}/{len(progress.learning_path)}[/dim]")
            else:
                console.print("[yellow]Could not load valid progress. Starting fresh.[/yellow]")
                progress = UserProgress()
    else:
        progress = UserProgress()
    
    try:
        # Skip Steps 1-3 if resuming
        if not resume_mode:
            # =================================================================
            # Step 1: Get passion and generate roles
            # =================================================================
            console.print("[bold cyan]Step 1: Discover Your Path[/bold cyan]")
            passion = console.input("\n[yellow]Enter your passion or interests:[/yellow] ").strip()
            
            if not passion:
                console.print("[red]No passion provided. Exiting.[/red]")
                return
            
            progress.passion = passion
            
            console.print("\n[dim]Analyzing job market and generating roles...[/dim]")
            with console.status("[bold green]Thinking...", spinner="dots"):
                roles_response = run_agent(
                    plan_agent1, 
                    f"Analyze the job market for someone passionate about: {passion}. Generate the top 3 roles."
                )
            
            console.print(Panel(Markdown(roles_response), title="📊 Market Analysis", border_style="green"))
            
            # Extract roles
            roles = extract_roles_from_response(roles_response)
            progress.available_roles = roles
            
            # =================================================================
            # Step 2: Select role
            # =================================================================
            console.print("\n[bold cyan]Step 2: Choose Your Role[/bold cyan]")
            role_idx = display_menu("Select a Role", roles)
            
            if role_idx == -1:
                console.print("[yellow]Exiting...[/yellow]")
                return
            
            selected_role = roles[role_idx]
            progress.selected_role = selected_role
            console.print(f"\n[green]✓ Selected:[/green] {selected_role}")
            
            # =================================================================
            # Step 3: Generate learning path
            # =================================================================
            console.print("\n[bold cyan]Step 3: Your Learning Path[/bold cyan]")
            
            with console.status("[bold green]Generating learning path...", spinner="dots"):
                path_response = run_agent(
                    plan_agent2,
                    f"Create a learning path for becoming a {selected_role}. List the key topics to master."
                )
            
            console.print(Panel(Markdown(path_response), title="🗺️ Learning Path", border_style="blue"))
            
            # Extract topics
            topics = extract_topics_from_response(path_response)
            progress.learning_path = topics
        else:
            # Resume mode - use stored learning path
            topics = progress.learning_path
            console.print(f"\n[bold cyan]Resuming: Learn and Test[/bold cyan]")
            console.print(f"[dim]Role: {progress.selected_role}[/dim]")
            console.print(f"[dim]Topics: {', '.join(topics)}[/dim]")
        
        # Outer loop to allow returning from Step 6 to Step 4
        while True:
            # =====================================================================
            # Step 4: Learning loop - select topics and take tests
            # =====================================================================
            console.print("\n[bold cyan]Step 4: Learn and Test[/bold cyan]")
            
            while True:
                remaining_topics = [t for t in topics if t not in progress.completed_topics]

                
                # Build menu options
                menu_options = remaining_topics.copy()
                
                # Add Challenge option if there are completed topics
                if progress.completed_topics:
                    menu_options.append("🔄 Challenge (Re-test completed topics)")
                
                menu_options.extend(["📊 View Progress", "🏁 Finish & Get Badge"])
                
                # Check if all topics done
                if not remaining_topics and not progress.completed_topics:
                    console.print("\n[yellow]No topics available.[/yellow]")
                    break
                elif not remaining_topics:
                    console.print("\n[green]🎉 All topics completed! You can still challenge yourself or finish.[/green]")
                
                topic_idx = display_menu("Select a Topic to Study", menu_options)
                
                if topic_idx == -1:
                    break
                
                selected_option = menu_options[topic_idx]
                
                if selected_option == "📊 View Progress":
                    if progress.test_results:
                        display_progress_table(progress)
                    else:
                        console.print("[dim]No tests completed yet.[/dim]")
                    continue
                elif selected_option == "🏁 Finish & Get Badge":
                    break
                elif selected_option == "🔄 Challenge (Re-test completed topics)":
                    # Show completed topics for challenge
                    challenge_idx = display_menu("Select a Topic to Challenge", progress.completed_topics)
                    if challenge_idx == -1:
                        continue
                    selected_topic = progress.completed_topics[challenge_idx]
                    is_challenge = True
                else:
                    # Selected a new topic
                    selected_topic = selected_option
                    is_challenge = False
                
                console.print(f"\n[bold]{'🔄 Challenge' if is_challenge else '📝 Test'} for: {selected_topic}[/bold]")
                
                with console.status("[bold green]Generating test...", spinner="dots"):
                    test_response = run_agent(
                        do_agent,
                        f"Generate a 5-question multiple choice test for role: {progress.selected_role}, topic: {selected_topic}. Questions must be specific to this role and topic combination."
                    )
                
                # Run interactive test
                result = parse_test_and_run(test_response, selected_topic)
                
                # Mark as challenge attempt if applicable
                if is_challenge:
                    result['is_challenge'] = True
                
                progress.test_results.append(result)
                
                if result['passed']:
                    if not is_challenge and selected_topic not in progress.completed_topics:
                        progress.completed_topics.append(selected_topic)
                    console.print(f"\n[green]✓ {'Challenge' if is_challenge else 'Topic'} '{selected_topic}' completed![/green]")
                else:
                    console.print(f"\n[yellow]You can retry this topic later.[/yellow]")

            
            # =====================================================================
            # Step 5: Generate badge and save results
            # =====================================================================
            if progress.test_results:
                console.print("\n[bold cyan]Step 5: Your Achievement[/bold cyan]")
                display_badge(progress)
                display_progress_table(progress)
                progress.save_to_json()
            
            # =====================================================================
            # Step 6: Final options - Community or Project
            # =====================================================================
            console.print("\n[bold cyan]Step 6: What's Next?[/bold cyan]")
            
            goto_topics = False
            while True:
                final_choice = display_menu("Choose Your Next Step", [
                    "🌐 Find Community",
                    "📋 Generate Project Plan",
                    "🔄 Back to Topics (test more)",
                    "🏅 View Badge",
                    "🚪 Exit"
                ], allow_exit=False)
                
                if final_choice == 0:  # Find Community
                    with console.status("[bold green]Finding communities...", spinner="dots"):
                        community_response = run_agent(
                            go_agent,
                            f"Find communities and networking opportunities for a {progress.selected_role} with skills in: {', '.join(progress.completed_topics)}"
                        )
                    console.print(Panel(Markdown(community_response), title="🌐 Community Resources", border_style="cyan"))
                
                elif final_choice == 1:  # Generate Project
                    with console.status("[bold green]Generating project plan...", spinner="dots"):
                        project_response = run_agent(
                            go_agent,
                            f"Generate a detailed project plan for a {progress.selected_role} to apply skills in: {', '.join(progress.completed_topics)}"
                        )
                    console.print(Panel(Markdown(project_response), title="📋 Project Plan", border_style="magenta"))
                    
                    # Save project plan
                    project_filename = "project_plan.md"
                    with open(project_filename, 'w') as f:
                        f.write(f"# Project Plan for {progress.selected_role}\n\n")
                        f.write(project_response)
                    console.print(f"[green]✓ Project plan saved to {project_filename}[/green]")
                
                elif final_choice == 2:  # Back to Topics
                    console.print("\n[cyan]Returning to topic selection...[/cyan]")
                    goto_topics = True
                    break
                
                elif final_choice == 3:  # View Badge
                    display_badge(progress)
                    display_progress_table(progress)
                
                else:  # Exit
                    goto_topics = False
                    break
            
            # If not going back to topics, exit the outer loop
            if not goto_topics:
                break

        
        console.print("\n[bold green]✅ Thank you for using Synthesis App![/bold green]")
        console.print("[dim]Check your Opik dashboard for detailed traces.[/dim]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error occurred:[/bold red] {str(e)}")
        console.print("[yellow]Please check your configuration and try again.[/yellow]")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure all traces are flushed to Opik
        console.print()
        with console.status("[cyan]Flushing traces to Opik...", spinner="dots"):
            opik_tracer.flush()
        console.print("[green]✓ Traces sent successfully![/green]")


if __name__ == "__main__":
    main()
