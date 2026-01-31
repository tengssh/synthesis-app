"""
Synthesis - Knowledge OS with Google ADK and Opik Integration
=======================================================

A multi-agent system for knowledge synthesis and research using Google's Agent Development Kit
with Comet Opik observability integration.

Effective Knowledge Management:
- Passion: bridge acquired knowledge to real-world applications, connect people with similar interests
- Market: analyze the current landscape, explore potential professional positions
- Capability: collect the most relevant topics and resources, create learning plans for better and comprehensive understanding

Three-step exploration:
1. Plan: 
    - (Passion) Text box with prompting interested topics, desired roles, or visions.
    - (Market) LLM generated possible jobs nano banana generated images with visual analysis on market share, salary, responsibilities, future development, etc.
    - (Capability) After user chooses one or more roles, generate a learning path using mindmap visualization.
2. Do:
    - According to learning paths, gamification using multiple choice tests at each node.
    - Generate learning status & statistics
3. Go:
    - Once each node’s grade beyond a criteria, create badges that can be shared with other people.
    - Facilitate users to apply skills on real-world projects.

Agents:
- PlanAgent1: Based on user's passion, perform analysis on current market and future trend, then generate the top 3 roles.
- PlanAgent2: Based on user's selected role, generate a learning path using mindmap visualization.
- DoAgent: Based on the generated learning path, generate multiple choice tests at each node with stored status.
- GoAgent: Based on the status of each learning path, generate badges that can be shared with other people and facilitate users to apply skills on real-world projects.
"""

import os
import asyncio
import warnings
import time
from functools import wraps
from dotenv import load_dotenv

# --- Opik Monkey-patch start ---
try:
    from typing import Optional
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

from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from opik.integrations.adk import OpikTracer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner

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
        "version": "1.0.0"
    },
    project_name="synthesis"
)


# Retry decorator for handling API overload errors
def retry_on_overload(max_retries=5, initial_delay=2, backoff_factor=2):
    """
    Decorator to retry function calls with exponential backoff on 503 errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
    """
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
                    
                    # Check if it's a 503 overload error
                    if "503" in error_str and ("overloaded" in error_str.lower() or "unavailable" in error_str.lower()):
                        if attempt < max_retries - 1:
                            console.print(f"[yellow]⚠️  API overloaded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})[/yellow]")
                            time.sleep(delay)
                            delay *= backoff_factor
                            continue
                    # If it's not a 503 error or we've exhausted retries, raise
                    raise
            
            # If we've exhausted all retries, raise the last exception
            raise last_exception
        return wrapper
    return decorator


# Define tools for PlanAgent1
def search_job_market(passion: str) -> str:
    """
    Search a job market for detailed information.

    Args:
        passion: The passion of the user

    Returns:
        Information about the job market
    """
    # PlanAgent1: Based on user's passion, perform analysis on current market and future trend, then generate the top 3 roles.
    return f"Current job market information for '{passion}': include market size, growth rate, and top companies."


def get_top_3_roles(market_info: str) -> str:
    """
    Get top 3 roles for the given market information.

    Args:
        market_info: The market information

    Returns:
        Top  3 roles for the given market information
    """
    # PlanAgent1: Based on user's passion, perform analysis on current market and future trend, then generate the top 3 roles.
    return f"Top 3 roles for '{market_info}': include job description, salary, and future development."


# Define tools for PlanAgent2
def search_learning_path(role: str) -> str:
    """
    Search a learning path for the given role.

    Args:
        role: The role of the user

    Returns:
        Learning path for the given role
    """
    # PlanAgent2: Based on user's selected role, generate a learning path using mindmap visualization.
    return f"Learning path for '{role}': include hierarchical structure and significance."


# Define tools for DoAgent
def generate_multiple_choice_tests(learning_path: str) -> str:
    """
    Generate multiple choice tests for the given learning path.

    Args:
        learning_path: The learning path

    Returns:
        Multiple choice tests for the given learning path
    """
    # DoAgent: Based on the generated learning path, generate multiple choice tests at each node with stored status.
    return f"Multiple choice tests for '{learning_path}': include questions, options, and correct answers."


def record_status(test_result: str) -> str:
    """
    Record the status of the test result.

    Args:
        test_result: The test result

    Returns:
        The status of the test result
    """
    # DoAgent: Based on the generated learning path, generate multiple choice tests at each node with stored status.
    return f"Status for '{test_result}': include correct/incorrect answers and score."


# Define tools for GoAgent
def generate_badges(test_result: str) -> str:
    """
    Generate badges for the given test result.

    Args:
        test_result: The test result

    Returns:
        Badges for the given test result
    """
    # GoAgent: Based on the status of each learning path, generate badges that can be shared with other people and facilitate users to apply skills on real-world projects.
    return f"Badges for '{test_result}': include badges that can be shared with other people and facilitate users to apply skills on real-world projects."


def generate_projects_markdown_plan(test_result: str) -> str:
    """
    Generate projects markdown plan for the given test result.

    Args:
        test_result: The test result

    Returns:
        Projects markdown plan for the given test result
    """
    # GoAgent: Based on the status of each learning path, generate projects that can be shared with other people and facilitate users to apply skills on real-world projects.
    return f"Projects markdown plan for '{test_result}': include projects that can be shared with other people and facilitate users to apply skills on real-world projects."


# Create the PlanAgent1 with tools
plan_agent1 = Agent(
    name="PlanAgent1",
    model="gemini-1.5-flash",
    instruction="""You are a creative job market analysis agent. Your role is to:
    1. Analyze the potential job market based on the provided passion
    2. Suggest TOP 3 creative and practical roles
    3. Provide accurate market analysis and practical role suggestions

    IMPORTANT: 
    - Do not ask follow-up questions. Always provide a complete suggestion immediately based on the passion given.
    - Use both available tools (search_job_market and get_top_3_roles) to gather comprehensive information
    - Be creative but practical, and consider common job market trends.""",
    output_key="market_analysis",
    tools=[search_job_market, get_top_3_roles],
    before_agent_callback=opik_tracer.before_agent_callback,
    after_agent_callback=opik_tracer.after_agent_callback,
    before_model_callback=opik_tracer.before_model_callback,
    after_model_callback=opik_tracer.after_model_callback,
    before_tool_callback=opik_tracer.before_tool_callback,
    after_tool_callback=opik_tracer.after_tool_callback,
)


# Create the PlanAgent2 with tools
plan_agent2 = Agent(
    name="PlanAgent2",
    model="gemini-1.5-flash",
    instruction="""You are a curriculum specialist. Your role is to:
    1. Look at the role that was selected
    2. Research additional context about that specific role using the available tools
    3. Provide comprehensive learning path using mindmap visualization
    4. Target topics with high demand and growth potential
    5. Provide practical learning path with actionable steps

    IMPORTANT:
    - Do not ask follow-up questions. Always provide complete research and information immediately.
    - Use available tools to gather comprehensive information
    - Provide your findings directly in a detailed format.""",
    output_key="learning_path",
    tools=[search_learning_path],
    before_agent_callback=opik_tracer.before_agent_callback,
    after_agent_callback=opik_tracer.after_agent_callback,
    before_model_callback=opik_tracer.before_model_callback,
    after_model_callback=opik_tracer.after_model_callback,
    before_tool_callback=opik_tracer.before_tool_callback,
    after_tool_callback=opik_tracer.after_tool_callback,
)

# Create the DoAgent with tools
do_agent = Agent(
    name="DoAgent",
    model="gemini-1.5-flash",
    instruction="""You are a professional test specialist. Your role is to:
    1. Look at the learning path that was selected
    2. Research additional context about that specific learning path using the available tools
    3. Provide comprehensive multiple choice tests for each node in the learning path
    4. Provide practical tests with actionable steps

    IMPORTANT:
    - Do not ask follow-up questions. Always provide complete research and information immediately.
    - Use available tools to gather comprehensive information
    - Provide your findings directly in a detailed format.""",
    output_key="test_result",
    tools=[generate_multiple_choice_tests, record_status],
    before_agent_callback=opik_tracer.before_agent_callback,
    after_agent_callback=opik_tracer.after_agent_callback,
    before_model_callback=opik_tracer.before_model_callback,
    after_model_callback=opik_tracer.after_model_callback,
    before_tool_callback=opik_tracer.before_tool_callback,
    after_tool_callback=opik_tracer.after_tool_callback,
)

# Create the GoAgent with tools
go_agent = Agent(
    name="GoAgent",
    model="gemini-1.5-flash",
    instruction="""You are a project specialist. Your role is to:
    1. Look at the test result that was selected
    2. Research additional context about that specific test result using the available tools
    3. Provide comprehensive projects markdown plan for each node in the test result
    4. Target topics with high demand and growth potential
    5. Provide practical projects markdown plan with actionable steps

    IMPORTANT:
    - Do not ask follow-up questions. Always provide complete research and information immediately.
    - Use available tools to gather comprehensive information
    - Provide your findings directly in a detailed format.""",
    output_key="projects_markdown_plan",
    tools=[generate_projects_markdown_plan],
    before_agent_callback=opik_tracer.before_agent_callback,
    after_agent_callback=opik_tracer.after_agent_callback,
    before_model_callback=opik_tracer.before_model_callback,
    after_model_callback=opik_tracer.after_model_callback,
    before_tool_callback=opik_tracer.before_tool_callback,
    after_tool_callback=opik_tracer.after_tool_callback,
)

# Create a Sequential Workflow Agent to ensure both agents run
plan_pipeline = SequentialAgent(
    name="PlanPipeline",
    sub_agents=[plan_agent1, plan_agent2],
    description="Executes plan suggestion followed by research in sequence."
)

# Create a Sequential Workflow Agent to ensure both agents run
do_pipeline = SequentialAgent(
    name="DoPipeline",
    sub_agents=[do_agent, go_agent],
    description="Executes test generation followed by project markdown plan generation in sequence."
)

# Create the Root Agent (orchestrator) that uses the sequential pipeline
root_agent = Agent(
    name="PlanMasterAgent",
    model="gemini-1.5-flash",
    instruction="""You are the Plan Master Agent. Your role is to coordinate plan creation.

    When a user provides passion:
    1. Delegate to the PlanPipeline agent which will automatically run both PlanAgent1 and PlanAgent2 in sequence
    2. The pipeline will provide you with both the market analysis (from PlanAgent1) and learning path (from PlanAgent2)
    3. Synthesize the information from both into a comprehensive, well-formatted final response

    Your final response should include:
    - Passion name and description
    - Complete market analysis
    - Top 3 roles
    - Learning path

    IMPORTANT: Do not ask follow-up questions. Always provide a complete, well-formatted response.""",
    sub_agents=[plan_pipeline, do_pipeline],
    before_agent_callback=opik_tracer.before_agent_callback,
    after_agent_callback=opik_tracer.after_agent_callback,
    before_model_callback=opik_tracer.before_model_callback,
    after_model_callback=opik_tracer.after_model_callback,
    before_tool_callback=opik_tracer.before_tool_callback,
    after_tool_callback=opik_tracer.after_tool_callback,
)


@retry_on_overload(max_retries=5, initial_delay=3, backoff_factor=2)
def run_agent_sync(user_prompt: str):
    """Synchronous function to run the agent with a user prompt."""

    async def _run():
        # Create runner with root agent
        runner = InMemoryRunner(
            agent=root_agent,
            app_name="synthesis-agent"
        )

        # Create session
        user_id = "user_1"
        session = await runner.session_service.create_session(
            app_name="synthesis-agent",
            user_id=user_id
        )

        # Create content message
        content = types.Content(
            role='user',
            parts=[types.Part(text=user_prompt)]
        )

        # Run the agent and collect response
        console.print()

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content
        ):
            # Print event content as it arrives with agent step indicators
            if hasattr(event, 'content') and event.content:
                if hasattr(event, 'author') and event.author:
                    author = event.author

                    # Print the content with rich formatting
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                # Determine panel style based on agent
                                if author == "PlanAgent1":
                                    title = "🍳 PlanAgent1"
                                    style = "green"
                                elif author == "PlanAgent2":
                                    title = "🔬 PlanAgent2"
                                    style = "blue"
                                elif author == "PlanMasterAgent":
                                    title = "📖 Final Plan"
                                    style = "bold magenta"
                                elif author == "DoAgent":
                                    title = "🍳 DoAgent"
                                    style = "green"
                                elif author == "GoAgent":
                                    title = "🔬 GoAgent"
                                    style = "blue"
                                elif author == "DoMasterAgent":
                                    title = "📖 Final Plan"
                                    style = "bold magenta"
                                else:
                                    title = author
                                    style = "white"

                                # Display as a panel
                                console.print(Panel(
                                    Markdown(part.text),
                                    title=title,
                                    border_style=style,
                                    padding=(1, 2)
                                ))

    # Run the async function in a new event loop
    asyncio.run(_run())


def main():
    """Main function to run the synthesis agent demo."""
    console.print(Panel.fit(
        "[bold cyan]Synthesis Agent Demo[/bold cyan]\n"
        "Powered by Google ADK with Opik Observability",
        border_style="cyan"
    ))
    console.print()

    # Get ingredients from user
    passion = console.input("[bold yellow]Enter passion (comma-separated):[/bold yellow] ").strip()

    if not passion:
        console.print("[red]No passion provided. Exiting.[/red]")
        return

    console.print()
    with console.status("[bold green]Preparing your plan...", spinner="dots"):
        pass

    # Create the user prompt
    user_prompt = f"I have this passion: {passion}. Please suggest a complete plan with all details, and include market research and background information."

    # Run the agent
    try:
        run_agent_sync(user_prompt)

        console.print()
        console.print("[bold green]✅ Done! Check your Opik dashboard for detailed traces.[/bold green]")

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
