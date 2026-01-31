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
        if choice == "0" and allow_exit:
            return -1
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        console.print("[red]Invalid choice. Please try again.[/red]")


def display_badge(progress: UserProgress):
    """Display achievement badge in terminal."""
    avg_score = progress.calculate_average_score()
    
    # Determine badge level
    if avg_score >= 90:
        badge_level = "🏆 GOLD"
        badge_color = "yellow"
    elif avg_score >= 70:
        badge_level = "🥈 SILVER"
        badge_color = "white"
    else:
        badge_level = "🥉 BRONZE"
        badge_color = "orange3"
    
    progress.badge_level = badge_level
    progress.badge_earned = True
    
    badge_content = f"""
[bold {badge_color}]
╔══════════════════════════════════════════════════╗
║           🎓 ACHIEVEMENT UNLOCKED 🎓              ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  Badge Level: {badge_level:<33} ║
║                                                  ║
║  Role: {progress.selected_role[:40]:<41} ║
║  Topics Completed: {len(progress.completed_topics):<28} ║
║  Average Score: {avg_score:.1f}%{' '*30} ║
║  Date: {progress.timestamp[:10]:<40} ║
║                                                  ║
╚══════════════════════════════════════════════════╝
[/bold {badge_color}]
"""
    console.print(Panel(badge_content, title="🏅 Your Badge", border_style=badge_color))


def display_progress_table(progress: UserProgress):
    """Display progress summary as a table."""
    table = Table(title="📊 Your Progress Summary", show_header=True, header_style="bold cyan")
    table.add_column("Topic", style="dim")
    table.add_column("Score", justify="center")
    table.add_column("Status", justify="center")
    
    for result in progress.test_results:
        status = "✅ Pass" if result['passed'] else "❌ Fail"
        score_str = f"{result['score']}/{result['total']}"
        table.add_row(result['topic'], score_str, status)
    
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

def create_agents():
    """Create and return all agents."""
    
    # PlanAgent1: Generate top 3 roles
    plan_agent1 = Agent(
        name="PlanAgent1",
        model="gemma-3-27b-it",
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
        
        Also include market analysis before the roles list.""",
        output_key="market_analysis",
        tools=[search_job_market, get_top_3_roles],
        before_agent_callback=opik_tracer.before_agent_callback,
        after_agent_callback=opik_tracer.after_agent_callback,
        before_model_callback=opik_tracer.before_model_callback,
        after_model_callback=opik_tracer.after_model_callback,
        before_tool_callback=opik_tracer.before_tool_callback,
        after_tool_callback=opik_tracer.after_tool_callback,
    )
    
    # PlanAgent2: Generate learning path
    plan_agent2 = Agent(
        name="PlanAgent2",
        model="gemma-3-27b-it",
        instruction="""You are a curriculum specialist. Based on the selected role:
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
        
        Include brief descriptions for each topic.""",
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
        model="gemma-3-27b-it",
        instruction="""You are a test specialist. Generate a multiple choice test.
        
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
        
        Make questions challenging but fair.""",
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
        model="gemma-3-27b-it",
        instruction="""You are a career development specialist. Based on the user's role and completed topics:
        
        For COMMUNITY requests:
        - List relevant online communities (Discord, Slack, Reddit)
        - Suggest professional networks and meetups
        - Recommend conferences and events
        
        For PROJECT requests:
        - Generate a detailed project plan in markdown format
        - Include project goals, milestones, and deliverables
        - Suggest technologies and resources needed
        
        Be specific and actionable.""",
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
        'passed': passed,
        'answers': user_answers
    }


# ============================================================================
# Main Interactive Workflow
# ============================================================================

def main():
    """Main function running the interactive multi-step workflow."""
    display_header()
    
    # Initialize progress tracking
    progress = UserProgress()
    
    # Create agents
    plan_agent1, plan_agent2, do_agent, go_agent = create_agents()
    
    try:
        # =====================================================================
        # Step 1: Get passion and generate roles
        # =====================================================================
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
        
        # =====================================================================
        # Step 2: Select role
        # =====================================================================
        console.print("\n[bold cyan]Step 2: Choose Your Role[/bold cyan]")
        role_idx = display_menu("Select a Role", roles)
        
        if role_idx == -1:
            console.print("[yellow]Exiting...[/yellow]")
            return
        
        selected_role = roles[role_idx]
        progress.selected_role = selected_role
        console.print(f"\n[green]✓ Selected:[/green] {selected_role}")
        
        # =====================================================================
        # Step 3: Generate learning path
        # =====================================================================
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
        
        # =====================================================================
        # Step 4: Learning loop - select topics and take tests
        # =====================================================================
        console.print("\n[bold cyan]Step 4: Learn and Test[/bold cyan]")
        
        while True:
            remaining_topics = [t for t in topics if t not in progress.completed_topics]
            
            if not remaining_topics:
                console.print("\n[green]🎉 Congratulations! You've completed all topics![/green]")
                break
            
            menu_options = remaining_topics + ["📊 View Progress", "🏁 Finish & Get Badge"]
            topic_idx = display_menu("Select a Topic to Study", menu_options)
            
            if topic_idx == -1:
                break
            elif menu_options[topic_idx] == "📊 View Progress":
                if progress.test_results:
                    display_progress_table(progress)
                else:
                    console.print("[dim]No tests completed yet.[/dim]")
                continue
            elif menu_options[topic_idx] == "🏁 Finish & Get Badge":
                break
            
            # Selected a topic - generate and run test
            selected_topic = remaining_topics[topic_idx]
            console.print(f"\n[bold]Preparing test for: {selected_topic}[/bold]")
            
            with console.status("[bold green]Generating test...", spinner="dots"):
                test_response = run_agent(
                    do_agent,
                    f"Generate a 5-question multiple choice test for the topic: {selected_topic}"
                )
            
            # Run interactive test
            result = parse_test_and_run(test_response, selected_topic)
            progress.test_results.append(result)
            
            if result['passed']:
                progress.completed_topics.append(selected_topic)
                console.print(f"\n[green]✓ Topic '{selected_topic}' completed![/green]")
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
        
        while True:
            final_choice = display_menu("Choose Your Next Step", [
                "🌐 Find Community",
                "📋 Generate Project Plan",
                "💾 Save Results Again",
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
            
            elif final_choice == 2:  # Save Results
                progress.save_to_json()
            
            else:  # Exit
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
