"""
Synthesis - Streamlit Web App
=============================

Single-page wizard with focused interface.
"""

import os
import json
import re
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List

import streamlit as st

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Synthesis - AI Career Explorer",
    page_icon="🧠",
    layout="centered",  # Centered for focused flow
    initial_sidebar_state="collapsed"
)

# ============================================================================
# Custom CSS (dark theme)
# ============================================================================

st.markdown("""
<style>
    /* Dark terminal-like styling */
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .step-complete {
        background: #1a1a2e;
        border-left: 3px solid #00d26a;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .step-current {
        background: #16213e;
        border-left: 3px solid #0ea5e9;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .analysis-box {
        background: #0f0f23;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    
    .role-button {
        width: 100%;
        text-align: left;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat-like message styling */
    .user-msg {
        background: #1e40af;
        padding: 0.75rem 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        display: inline-block;
    }
    
    .bot-msg {
        background: #1f2937;
        padding: 0.75rem 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
    }
    
    /* Progress indicator */
    .progress-step {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    .progress-done { background: #166534; color: white; }
    .progress-current { background: #0369a1; color: white; }
    .progress-pending { background: #374151; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class UserProgress:
    passion: str = ""
    available_roles: List[str] = field(default_factory=list)
    selected_role: str = ""
    learning_path: List[str] = field(default_factory=list)
    completed_topics: List[str] = field(default_factory=list)
    test_results: List[dict] = field(default_factory=list)
    badge_earned: bool = False
    badge_level: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    market_analysis: str = ""
    learning_response: str = ""
    community_response: str = ""
    project_response: str = ""
    
    def calculate_average_score(self) -> float:
        if not self.test_results:
            return 0.0
        total_score = sum(r['score'] for r in self.test_results)
        total_possible = sum(r['total'] for r in self.test_results)
        return (total_score / total_possible * 100) if total_possible > 0 else 0.0
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    def get_step(self) -> int:
        """Current step in the flow."""
        if not self.passion:
            return 1
        if not self.selected_role:
            return 2
        if not self.learning_path:
            return 3
        if len(self.completed_topics) < len(self.learning_path):
            return 4
        return 5  # Complete


# ============================================================================
# Session State
# ============================================================================

def init_session_state():
    defaults = {
        'progress': UserProgress(),
        'api_key_valid': False,
        'current_questions': [],
        'current_topic': '',
        'opik_tracer': None,
        'messages': [],  # Chat history
        'agents_loaded': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ============================================================================
# Lazy Loading
# ============================================================================

@st.cache_resource
def load_agents_module():
    """Lazy load heavy dependencies."""
    import warnings
    warnings.filterwarnings('ignore')
    
    from google.adk.agents import Agent
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    
    return Agent, InMemoryRunner, types


def get_opik_tracer():
    """Get or create Opik tracer if API key is available."""
    if st.session_state.opik_tracer:
        return st.session_state.opik_tracer
    opik_key = os.getenv("OPIK_API_KEY")
    if opik_key:
        try:
            from opik.integrations.adk import OpikTracer
            st.session_state.opik_tracer = OpikTracer(
                name="synthesis-web", tags=["synthesis"], project_name="synthesis"
            )
        except:
            pass
    return st.session_state.opik_tracer


# ============================================================================
# Tool Functions
# ============================================================================

def search_job_market(passion: str) -> str:
    return f"Analyzed job market for passion: {passion}"

def get_top_3_roles(passion: str) -> str:
    return f"Generated top 3 roles for: {passion}"

def search_learning_path(role: str) -> str:
    return f"Generated learning path for: {role}"

def generate_test_questions(topic: str) -> str:
    return f"Generated test questions for: {topic}"

def find_community_resources(role: str, skills: list) -> str:
    return f"Found communities for '{role}'."

def generate_project_ideas(role: str, skills: list) -> str:
    return f"Generated project ideas for '{role}'."


# ============================================================================
# Agent Creation & Execution
# ============================================================================

def create_agents(tracer=None):
    Agent, _, _ = load_agents_module()
    
    callbacks = {}
    if tracer:
        callbacks = {
            'before_agent_callback': tracer.before_agent_callback,
            'after_agent_callback': tracer.after_agent_callback,
            'before_model_callback': tracer.before_model_callback,
            'after_model_callback': tracer.after_model_callback,
            'before_tool_callback': tracer.before_tool_callback,
            'after_tool_callback': tracer.after_tool_callback,
        }
    
    plan_agent1 = Agent(
        name="PlanAgent1", model="gemma-3-27b-it",
        instruction="""You are a job market analysis agent. Based on the user's passion:
        1. Analyze the job market
        2. Generate EXACTLY 3 specific job roles
        
        EACH ROLE MUST INCLUDE:
        - Role name
        - Brief description (1-2 sentences)
        - Salary range
        - Future growth/development opportunities
        
        OUTPUT FORMAT:
        TOP 3 ROLES:
        1. [Role] - Description. Salary: [range]. Future: [growth]
        2. [Role] - Description. Salary: [range]. Future: [growth]
        3. [Role] - Description. Salary: [range]. Future: [growth]""",
        output_key="market_analysis",
        tools=[search_job_market, get_top_3_roles], **callbacks
    )
    
    plan_agent2 = Agent(
        name="PlanAgent2", model="gemma-3-27b-it",
        instruction="""You are a learning path designer. Create 5-7 numbered topics.
        
OUTPUT:
KEY TOPICS:
1. [Topic] - Brief description
2. [Topic] - Brief description
...

Each topic builds on previous ones. Be practical and actionable.""",
        output_key="learning_path",
        tools=[search_learning_path], **callbacks
    )
    
    do_agent = Agent(
        name="DoAgent", model="gemma-3-27b-it",
        instruction="""Generate exactly 5 multiple choice questions.
        
FORMAT:
Q1: [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct: [A/B/C/D]

Questions must be specific to the role and topic. No placeholders.""",
        output_key="test_questions",
        tools=[generate_test_questions], **callbacks
    )
    
    go_agent = Agent(
        name="GoAgent", model="gemma-3-27b-it",
        instruction="""Career development specialist.
        
For COMMUNITY: List Discord servers, subreddits, Slack groups with specific names.
For PROJECT: Generate detailed project plan with goals and milestones.

Do NOT ask follow-up questions. Give complete recommendations.""",
        output_key="career_guidance",
        tools=[find_community_resources, generate_project_ideas], **callbacks
    )
    
    return plan_agent1, plan_agent2, do_agent, go_agent


def run_agent(agent, prompt: str) -> str:
    """Run agent and return response."""
    _, InMemoryRunner, types = load_agents_module()
    
    response_text = ""
    
    async def _run():
        nonlocal response_text
        runner = InMemoryRunner(agent=agent, app_name="synthesis-web")
        session = await runner.session_service.create_session(app_name="synthesis-web", user_id="web_user")
        content = types.Content(role='user', parts=[types.Part(text=prompt)])
        
        async for event in runner.run_async(user_id="web_user", session_id=session.id, new_message=content):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text + "\n"
    
    asyncio.run(_run())
    return response_text


# ============================================================================
# Helper Functions
# ============================================================================


def extract_roles(response: str) -> List[str]:
    roles = []
    for line in response.split('\n'):
        line = line.strip()
        if re.match(r'^[1-3][\.\)]\s+', line):
            role = re.sub(r'^[1-3][\.\)]\s+', '', line)
            if ' - ' in role:
                role = role.split(' - ')[0].strip()
            if role:
                roles.append(role)
    return roles[:3] if roles else ["Role 1", "Role 2", "Role 3"]


def extract_topics(response: str) -> List[str]:
    topics = []
    for line in response.split('\n'):
        line = line.strip()
        if re.match(r'^[1-7][\.\)]\s+', line):
            topic = re.sub(r'^[1-7][\.\)]\s+', '', line)
            if ':' in topic:
                topic = topic.split(':')[0].strip()
            elif ' - ' in topic:
                topic = topic.split(' - ')[0].strip()
            if topic:
                topics.append(topic)
    return topics[:7] if topics else ["Fundamentals", "Core Concepts", "Advanced Topics"]


def parse_questions(response: str, topic: str) -> List[dict]:
    questions = []
    current_q = None
    options = {}
    correct = None
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('Q') and ':' in line:
            if current_q and options and correct:
                questions.append({'question': current_q, 'options': options.copy(), 'correct': correct})
            current_q = line.split(':', 1)[1].strip()
            options = {}
            correct = None
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            options[line[0]] = line[2:].strip()
        elif line.lower().startswith('correct:'):
            correct = line.split(':')[1].strip().upper()
    
    if current_q and options and correct:
        questions.append({'question': current_q, 'options': options.copy(), 'correct': correct})
    
    return questions if questions else [
        {'question': f'What is key in {topic}?', 'options': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}, 'correct': 'A'}
    ]


def get_badge_info(progress: UserProgress) -> tuple:
    avg = progress.calculate_average_score()
    completion = len(progress.completed_topics) / len(progress.learning_path) if progress.learning_path else 0
    if avg >= 90 and completion >= 0.8:
        return "🏆 GOLD", "#FFD700"
    elif avg >= 70 and completion >= 0.5:
        return "🥈 SILVER", "#C0C0C0"
    return "🥉 BRONZE", "#CD7F32"


# ============================================================================
# Progress Indicator
# ============================================================================

def render_progress():
    """Render horizontal progress steps."""
    p = st.session_state.progress
    step = p.get_step()
    
    steps = ["API Key", "Passion", "Role", "Learn", "Go"]
    
    html = '<div style="text-align: center; margin: 1rem 0;">'
    for i, s in enumerate(steps):
        if i == 0:  # API step always done if we're here
            cls = "progress-done"
        elif i < step:
            cls = "progress-done"
        elif i == step:
            cls = "progress-current"
        else:
            cls = "progress-pending"
        html += f'<span class="progress-step {cls}">{s}</span>'
        if i < len(steps) - 1:
            html += ' → '
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# Wizard Steps
# ============================================================================

def render_step_api():
    """Step 0: API Key entry."""
    st.title("🧠 Synthesis")
    st.caption("AI-Powered Career Exploration")
    
    st.markdown("---")
    
    api_key = st.text_input(
        "🔑 Enter your Google Gemini API Key",
        type="password",
        placeholder="Paste API key to begin...",
        help="Get free: aistudio.google.com/apikey"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start", type="primary", use_container_width=True, disabled=not api_key):
            os.environ["GOOGLE_API_KEY"] = api_key
            st.session_state.api_key_valid = True
            st.rerun()
    with col2:
        uploaded = st.file_uploader("📂 Resume", type=['json'], label_visibility="collapsed")
        if uploaded and api_key:
            try:
                os.environ["GOOGLE_API_KEY"] = api_key
                st.session_state.api_key_valid = True
                st.session_state.progress = UserProgress(**json.load(uploaded))
                st.rerun()
            except:
                st.error("Invalid file")


def render_step_passion():
    """Step 1: Enter passion."""
    p = st.session_state.progress
    
    st.markdown("### 💡 What's your passion?")
    st.caption("Enter an interest area and I'll find matching career paths.")
    
    passion = st.text_input(
        "Your interest",
        placeholder="e.g., AI, music production, game design, cooking...",
        label_visibility="collapsed"
    )
    
    if st.button("🔍 Analyze Market", type="primary", disabled=not passion):
        with st.status("Analyzing job market...", expanded=True) as status:
            st.write("🔄 Connecting to Gemini...")
            st.write("📊 Analyzing market trends...")
            tracer = get_opik_tracer()
            plan_agent1, _, _, _ = create_agents(tracer)
            
            st.write("🎯 Finding matching roles...")
            response = run_agent(plan_agent1, f"My passion is: {passion}")
            
            p.passion = passion
            p.market_analysis = response
            p.available_roles = extract_roles(response)
            
            status.update(label="✅ Analysis complete!", state="complete")
        
        st.rerun()


def render_step_roles():
    """Step 2: Select role."""
    p = st.session_state.progress
    
    # Show completed step
    st.markdown(f'<div class="step-complete">✅ Passion: <strong>{p.passion}</strong></div>', unsafe_allow_html=True)
    
    # Market analysis - format with line breaks
    st.markdown("### 📊 Market Analysis")
    # Convert to proper markdown with line breaks
    formatted_analysis = p.market_analysis.replace('1.', '\n\n**1.**').replace('2.', '\n\n**2.**').replace('3.', '\n\n**3.**')
    st.markdown(formatted_analysis)
    
    st.markdown("### 🎯 Choose Your Path")
    st.caption("Select a role to generate your learning path.")
    
    for i, role in enumerate(p.available_roles):
        if st.button(f"👉 {role}", key=f"role_{i}", use_container_width=True):
            with st.status(f"Creating learning path for {role}...", expanded=True) as status:
                st.write("📚 Designing curriculum...")
                tracer = get_opik_tracer()
                _, plan_agent2, _, _ = create_agents(tracer)
                
                st.write("🎓 Generating topics...")
                response = run_agent(plan_agent2, f"Create learning path for: {role}")
                
                p.selected_role = role
                p.learning_response = response
                p.learning_path = extract_topics(response)
                
                status.update(label="✅ Learning path ready!", state="complete")
            
            st.rerun()


def render_step_learn():
    """Step 3: Learning and testing."""
    p = st.session_state.progress
    
    # Completed steps (collapsed)
    with st.expander("✅ Previous Steps", expanded=False):
        st.write(f"**Passion:** {p.passion}")
        st.write(f"**Role:** {p.selected_role}")
    
    # Progress
    completed = len(p.completed_topics)
    total = len(p.learning_path)
    
    st.markdown(f"### 📝 Learning Progress: {completed}/{total}")
    st.progress(completed / total if total > 0 else 0)
    
    # Topics list with retry option for failed
    remaining = [t for t in p.learning_path if t not in p.completed_topics]
    failed_topics = []
    
    for topic in p.learning_path:
        if topic in p.completed_topics:
            result = next((r for r in p.test_results if r['topic'] == topic), None)
            if result:
                icon = "✅" if result['passed'] else "❌"
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{icon} **{topic}** — Score: {result['score']}/{result['total']}")
                with col2:
                    if not result['passed']:
                        failed_topics.append(topic)
                        if st.button("🔄 Retry", key=f"retry_{topic}"):
                            # Remove from completed to allow retry
                            p.completed_topics.remove(topic)
                            p.test_results = [r for r in p.test_results if r['topic'] != topic]
                            st.rerun()
        else:
            st.info(f"○ {topic}")
    
    # Show test options
    test_options = remaining.copy()
    
    if test_options:
        st.markdown("---")
        st.markdown("### 🧪 Take a Test")
        
        topic = st.selectbox("Select topic", test_options, label_visibility="collapsed")
        
        if st.button("📝 Start Test", type="primary"):
            with st.status(f"Generating test for {topic}...", expanded=True) as status:
                st.write("🧠 Creating questions...")
                tracer = get_opik_tracer()
                _, _, do_agent, _ = create_agents(tracer)
                
                response = run_agent(do_agent, f"Role: {p.selected_role}\nTopic: {topic}\nGenerate test.")
                
                st.session_state.current_topic = topic
                st.session_state.current_questions = parse_questions(response, topic)
                
                status.update(label="✅ Test ready!", state="complete")
            
            st.rerun()
    
    # If test is active
    if st.session_state.current_questions and st.session_state.current_topic:
        st.markdown("---")
        st.markdown(f"### Test: {st.session_state.current_topic}")
        
        questions = st.session_state.current_questions
        
        with st.form("test_form"):
            answers = {}
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}:** {q['question']}")
                answers[i] = st.radio(
                    f"Q{i+1}",
                    options=['A', 'B', 'C', 'D'],
                    format_func=lambda x, q=q: f"{x}) {q['options'].get(x, '')}",
                    key=f"q_{i}",
                    horizontal=False,  # Vertical layout
                    label_visibility="collapsed",
                    index=None  # No default selection
                )
            
            if st.form_submit_button("✅ Submit", type="primary"):
                # Check if all questions answered
                unanswered = [i+1 for i, a in answers.items() if a is None]
                if unanswered:
                    st.error(f"Please answer all questions. Missing: Q{', Q'.join(map(str, unanswered))}")
                else:
                    score = sum(1 for i, q in enumerate(questions) if answers.get(i) == q['correct'])
                    total = len(questions)
                    passed = score >= total * 0.6
                    
                    p.test_results.append({
                        'topic': st.session_state.current_topic,
                        'score': score,
                        'total': total,
                        'passed': passed
                    })
                    p.completed_topics.append(st.session_state.current_topic)
                    
                    st.session_state.current_topic = ''
                    st.session_state.current_questions = []
                    
                    if passed:
                        st.success(f"🎉 Passed! {score}/{total}")
                    else:
                        st.warning(f"Score: {score}/{total}. Keep practicing!")
                    
                    st.rerun()
    
    if not remaining:
        st.markdown("---")
        st.success("🎉 All topics completed!")
        if st.button("🚀 See Results", type="primary"):
            st.rerun()


def render_step_go():
    """Step 4: Results and badge."""
    p = st.session_state.progress
    
    badge_level, badge_color = get_badge_info(p)
    avg = p.calculate_average_score()
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {badge_color}40, {badge_color}20); 
                border: 2px solid {badge_color}; border-radius: 20px; padding: 2rem; 
                text-align: center; margin: 1rem 0;">
        <h1 style="font-size: 4rem; margin: 0;">{badge_level}</h1>
        <h2>{p.selected_role}</h2>
        <p>Score: {avg:.1f}% | Topics: {len(p.completed_topics)}/{len(p.learning_path)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Test history
    if p.test_results:
        st.markdown("### 📊 Test Results")
        for r in p.test_results:
            icon = "✅" if r['passed'] else "❌"
            st.write(f"{icon} **{r['topic']}**: {r['score']}/{r['total']}")
    
    st.markdown("---")
    
    # Next steps
    st.markdown("### 🚀 Next Steps")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌐 Find Community", use_container_width=True):
            with st.spinner("Finding communities..."):
                tracer = get_opik_tracer()
                _, _, _, go_agent = create_agents(tracer)
                p.community_response = run_agent(go_agent, 
                    f"Role: {p.selected_role}\nSkills: {', '.join(p.completed_topics)}\nFind communities.")
            st.rerun()
    
    with col2:
        if st.button("📋 Get Project Ideas", use_container_width=True):
            with st.spinner("Generating project..."):
                tracer = get_opik_tracer()
                _, _, _, go_agent = create_agents(tracer)
                p.project_response = run_agent(go_agent,
                    f"Role: {p.selected_role}\nSkills: {', '.join(p.completed_topics)}\nGenerate project plan.")
            st.rerun()
    
    if p.community_response:
        with st.expander("🌐 Community Resources", expanded=True):
            st.markdown(p.community_response)
    
    if p.project_response:
        with st.expander("📋 Project Plan", expanded=True):
            st.markdown(p.project_response)
    
    st.markdown("---")
    
    # Export
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("💾 Export Progress", p.to_json(), "synthesis_progress.json", use_container_width=True)
    with col2:
        badge_text = f"🧠 SYNTHESIS: {badge_level}\n{p.selected_role}\nScore: {avg:.1f}%"
        st.download_button("🏅 Share Badge", badge_text, "synthesis_badge.txt", use_container_width=True)
    with col3:
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.progress = UserProgress()
            st.session_state.api_key_valid = False
            st.rerun()


# ============================================================================
# Main App
# ============================================================================

def main():
    # API key check
    if not st.session_state.api_key_valid:
        render_step_api()
        return
    
    p = st.session_state.progress
    step = p.get_step()
    
    # Header
    st.title("🧠 Synthesis")
    render_progress()
    st.markdown("---")
    
    # Wizard steps
    if step == 1:
        render_step_passion()
    elif step == 2:
        render_step_roles()
    elif step == 3 or step == 4:
        render_step_learn()
    else:
        render_step_go()


if __name__ == "__main__":
    main()
