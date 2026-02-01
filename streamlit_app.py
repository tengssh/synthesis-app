"""
Synthesis - Streamlit Web App (Enhanced UI/UX)
==============================================

Web interface for the Synthesis Knowledge OS using Streamlit.
Features: Mobile-friendly, navigation tabs, export/resume, shareable badge.
"""

import os
import asyncio
import warnings
import time
import json
import re
import base64
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# --- Opik Monkey-patch start ---
try:
    from opik.llm_usage import google_usage
    from pydantic import Field
    from pydantic.fields import FieldInfo
    
    if hasattr(google_usage, "GoogleGeminiUsage"):
        google_usage.GoogleGeminiUsage.model_fields['candidates_token_count'] = FieldInfo(
            annotation=Optional[int], 
            default=None,
            description="Number of tokens in the response(s)."
        )
        google_usage.GoogleGeminiUsage.model_rebuild(force=True)
except Exception:
    pass
# --- Opik Monkey-patch end ---

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Synthesis - AI Career Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapsed for mobile
)

# ============================================================================
# Custom CSS for Mobile-Friendly UI
# ============================================================================

st.markdown("""
<style>
    /* Mobile-friendly adjustments */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        border-radius: 10px;
        margin: 0.25rem 0;
    }
    
    /* Navigation tabs */
    .nav-tabs {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        padding: 1rem 0;
        flex-wrap: wrap;
    }
    
    .nav-tab {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    /* Badge card styling */
    .badge-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    .badge-gold { background: linear-gradient(135deg, #f5af19 0%, #f12711 100%); }
    .badge-silver { background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%); }
    .badge-bronze { background: linear-gradient(135deg, #cd7f32 0%, #8b4513 100%); }
    
    /* Progress bar */
    .progress-container {
        background: #e0e0e0;
        border-radius: 10px;
        height: 10px;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Compact metrics for mobile */
    @media (max-width: 768px) {
        .stMetric {
            padding: 0.5rem !important;
        }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.25rem !important; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
    
    def calculate_average_score(self) -> float:
        if not self.test_results:
            return 0.0
        total_score = sum(r['score'] for r in self.test_results)
        total_possible = sum(r['total'] for r in self.test_results)
        return (total_score / total_possible * 100) if total_possible > 0 else 0.0
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UserProgress':
        data = json.loads(json_str)
        return cls(**data)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'page': 'home',  # home, plan, do, go, profile
        'progress': UserProgress(),
        'api_key_valid': False,
        'current_questions': [],
        'current_topic': '',
        'opik_tracer': None,
        'show_test': False,
        'community_response': '',
        'project_response': ''
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


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
    skills_str = ", ".join(skills) if skills else "general"
    return f"Found communities for '{role}' with skills in {skills_str}."

def generate_project_ideas(role: str, skills: list) -> str:
    skills_str = ", ".join(skills) if skills else "general"
    return f"Generated project ideas for '{role}' using skills: {skills_str}."


# ============================================================================
# Agent Creation (Cached for Performance)
# ============================================================================

@st.cache_resource
def get_cached_agents():
    """Create and cache agents for better performance."""
    return create_agents_internal(None)

def create_agents_internal(tracer=None):
    """Create and return all agents."""
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
        1. Analyze the job market  2. Generate EXACTLY 3 specific job roles
        OUTPUT FORMAT: Return "TOP 3 ROLES:" followed by exactly 3 numbered roles.
        Example: TOP 3 ROLES:
        1. [Role] - Brief description
        2. [Role] - Brief description
        3. [Role] - Brief description
        Also include market analysis before the roles list.
        STYLE: Be direct and user-friendly.""",
        output_key="market_analysis",
        tools=[search_job_market, get_top_3_roles], **callbacks
    )
    
    plan_agent2 = Agent(
        name="PlanAgent2", model="gemma-3-27b-it",
        instruction="""You are an expert learning path designer.
        OUTPUT STRUCTURE: KEY TOPICS: [5-7 numbered topics with brief descriptions]
        RULES: Each topic must build on previous ones. Include theoretical and practical elements.""",
        output_key="learning_path",
        tools=[search_learning_path], **callbacks
    )
    
    do_agent = Agent(
        name="DoAgent", model="gemma-3-27b-it",
        instruction="""Generate a multiple choice test. OUTPUT FORMAT:
        Q1: [Question]  A) B) C) D) options  Correct: [A/B/C/D]
        Generate exactly 5 questions relevant to the role and topic.""",
        output_key="test_questions",
        tools=[generate_test_questions], **callbacks
    )
    
    go_agent = Agent(
        name="GoAgent", model="gemma-3-27b-it",
        instruction="""Career development specialist. For COMMUNITY: List relevant online communities, networks, events.
        For PROJECT: Generate detailed project plan with goals, milestones, technologies.
        IMPORTANT: Provide complete, actionable information. Do NOT ask follow-up questions.""",
        output_key="career_guidance",
        tools=[find_community_resources, generate_project_ideas], **callbacks
    )
    
    return plan_agent1, plan_agent2, do_agent, go_agent

def create_agents(tracer=None):
    """Get agents - use cached if no tracer, else create new."""
    if tracer is None:
        return get_cached_agents()
    return create_agents_internal(tracer)


# ============================================================================
# Agent Runner
# ============================================================================

def run_agent(agent: Agent, prompt: str) -> str:
    """Run a single agent and return the response text."""
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

def extract_roles_from_response(response: str) -> List[str]:
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


def extract_topics_from_response(response: str) -> List[str]:
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
            current_q = line.split(':', 1)[1].strip() if ':' in line else line
            options = {}
            correct = None
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            options[line[0]] = line[2:].strip()
        elif line.lower().startswith('correct:'):
            correct = line.split(':')[1].strip().upper()
    
    if current_q and options and correct:
        questions.append({'question': current_q, 'options': options.copy(), 'correct': correct})
    
    if not questions:
        questions = [
            {'question': f'What is a key concept in {topic}?', 'options': {'A': 'Option A', 'B': 'Option B', 'C': 'Option C', 'D': 'Option D'}, 'correct': 'A'},
            {'question': f'How is {topic} applied?', 'options': {'A': 'Method A', 'B': 'Method B', 'C': 'Method C', 'D': 'Method D'}, 'correct': 'B'},
        ]
    return questions


def get_badge_info(progress: UserProgress) -> tuple:
    avg_score = progress.calculate_average_score()
    completion = len(progress.completed_topics) / len(progress.learning_path) if progress.learning_path else 0
    
    if avg_score >= 90 and completion >= 0.8:
        return "🏆 GOLD", "badge-gold", "#FFD700"
    elif avg_score >= 70 and completion >= 0.5:
        return "🥈 SILVER", "badge-silver", "#C0C0C0"
    else:
        return "🥉 BRONZE", "badge-bronze", "#CD7F32"


def generate_shareable_badge(progress: UserProgress) -> str:
    """Generate shareable badge text."""
    badge_level, _, _ = get_badge_info(progress)
    avg_score = progress.calculate_average_score()
    
    badge_text = f"""
🧠 SYNTHESIS ACHIEVEMENT 🧠
━━━━━━━━━━━━━━━━━━━━━━
{badge_level} Badge Earned!

🎯 Role: {progress.selected_role}
📚 Topics: {len(progress.completed_topics)}/{len(progress.learning_path)}
📊 Score: {avg_score:.1f}%
📅 Date: {progress.timestamp[:10]}
━━━━━━━━━━━━━━━━━━━━━━
Try it: synthesis-app.streamlit.app
"""
    return badge_text


# ============================================================================
# Navigation
# ============================================================================

def render_navigation():
    """Render top navigation bar."""
    p = st.session_state.progress
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🧠 Synthesis")
    with col2:
        if st.session_state.api_key_valid:
            st.caption(f"✓ API Ready")
    
    # Navigation buttons
    if st.session_state.api_key_valid:
        cols = st.columns(5)
        
        nav_items = [
            ("🏠", "home", "Home"),
            ("📋", "plan", "Plan"),
            ("📝", "do", "Do"),
            ("🚀", "go", "Go"),
            ("👤", "profile", "Profile")
        ]
        
        for i, (icon, page, label) in enumerate(nav_items):
            with cols[i]:
                btn_type = "primary" if st.session_state.page == page else "secondary"
                if st.button(f"{icon} {label}", key=f"nav_{page}", use_container_width=True, type=btn_type):
                    st.session_state.page = page
                    st.rerun()
    
    st.divider()


# ============================================================================
# Pages
# ============================================================================

def render_home():
    """Home page - clean and minimal."""
    st.title("🧠 Synthesis")
    st.caption("AI-Powered Career Exploration")
    
    st.divider()
    
    # Compact API + Actions row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        api_key = st.text_input("🔑 Google Gemini API Key", type="password", 
                                placeholder="Paste your API key here",
                                help="Get free: aistudio.google.com/apikey")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ.pop("GEMINI_API_KEY", None)
            st.session_state.api_key_valid = True
    
    with col2:
        st.write("")  # Spacer
        if st.session_state.api_key_valid:
            st.success("✓ Ready")
        else:
            st.info("Enter key →")
    
    # Action buttons in a row
    if st.session_state.api_key_valid:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start New", type="primary", use_container_width=True):
                st.session_state.progress = UserProgress()
                st.session_state.page = 'plan'
                st.rerun()
        with col2:
            uploaded = st.file_uploader("📂 Resume", type=['json'], label_visibility="collapsed")
            if uploaded:
                try:
                    st.session_state.progress = UserProgress(**json.load(uploaded))
                    st.session_state.page = 'plan'
                    st.rerun()
                except:
                    st.error("Invalid file")
    
    st.divider()
    
    # Compact steps - single line
    st.markdown("**📋 Plan** → **📝 Do** → **🚀 Go** → **🏅 Earn**")
    st.caption("Discover roles • Learn & test • Find community • Get your badge")
    
    st.divider()
    st.caption("⚠️ AI can make mistakes. Verify recommendations.")





def render_venn_diagram():
    """Render the Passion/Skill/Market Venn diagram."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('white')
    
    v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), 
              set_labels=('Passion', 'Skill', 'Market'),
              ax=ax)
    
    # Customize colors
    if v.get_patch_by_id('100'): v.get_patch_by_id('100').set_color('#FFCCCC')
    if v.get_patch_by_id('010'): v.get_patch_by_id('010').set_color('#CCFFCC')
    if v.get_patch_by_id('001'): v.get_patch_by_id('001').set_color('#CCECFF')
    if v.get_patch_by_id('110'): v.get_patch_by_id('110').set_color('#FFEEAA')
    if v.get_patch_by_id('101'): v.get_patch_by_id('101').set_color('#EECCFF')
    if v.get_patch_by_id('011'): v.get_patch_by_id('011').set_color('#AAFFEE')
    if v.get_patch_by_id('111'): v.get_patch_by_id('111').set_color('#AADDAA')
    
    # Set labels
    labels = {
        '100': '', '010': '', '001': '',
        '110': 'No money', '101': 'Incapable', '011': 'Unhappy',
        '111': 'Ideal Job'
    }
    for label_id, text in labels.items():
        label = v.get_label_by_id(label_id)
        if label:
            label.set_text(text)
            if label_id == '111':
                label.set_fontsize(12)
                label.set_fontweight('bold')
    
    # Style set labels
    for text in v.set_labels:
        if text:
            text.set_fontsize(14)
            text.set_fontweight('bold')
    
    ax.set_title('Find Your Ideal Career', fontsize=16, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig


def render_plan():
    """Plan page - Passion entry and role selection."""
    st.header("📋 PLAN: Discover Your Path")
    
    p = st.session_state.progress
    
    # Step 1: Enter passion
    if not p.passion or not p.available_roles:
        # Show Venn diagram
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(render_venn_diagram())
        with col2:
            st.markdown("### Find where your passion meets opportunity!")
            st.write("We'll analyze the market to find roles that match your interests and have good career potential.")
        
        st.divider()
        st.subheader("What's Your Passion?")
        passion = st.text_input("Enter your interest", placeholder="AI, music, gaming, cooking...")
        
        if st.button("🔍 Find Career Paths", type="primary", disabled=not passion):
            with st.spinner("Analyzing job market..."):
                try:
                    plan_agent1, _, _, _ = create_agents(st.session_state.opik_tracer)
                    response = run_agent(plan_agent1, f"My passion is: {passion}")
                    
                    p.passion = passion
                    p.market_analysis = response
                    p.available_roles = extract_roles_from_response(response)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Step 2: Show market analysis and select role
    elif not p.selected_role:
        st.success(f"Passion: **{p.passion}**")
        
        with st.expander("📊 Market Analysis", expanded=True):
            st.markdown(p.market_analysis)
        
        st.subheader("Choose Your Role")
        for i, role in enumerate(p.available_roles):
            if st.button(f"👉 {role}", key=f"role_{i}", use_container_width=True):
                with st.spinner(f"Creating learning path..."):
                    try:
                        _, plan_agent2, _, _ = create_agents(st.session_state.opik_tracer)
                        response = run_agent(plan_agent2, f"Create learning path for: {role}")
                        
                        p.selected_role = role
                        p.learning_response = response
                        p.learning_path = extract_topics_from_response(response)
                        st.session_state.page = 'do'
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Already selected - show summary
    else:
        st.success(f"✓ Passion: **{p.passion}**")
        st.success(f"✓ Role: **{p.selected_role}**")
        
        with st.expander("📚 Your Learning Path"):
            st.markdown(p.learning_response)
        
        if st.button("🔄 Start Fresh", type="secondary"):
            st.session_state.progress = UserProgress()
            st.rerun()


def render_do():
    """Do page - Learning path and tests."""
    st.header("📝 DO: Learn & Test")
    
    p = st.session_state.progress
    
    if not p.selected_role:
        st.info("👈 Go to PLAN first to select your career path")
        return
    
    # Progress overview
    total_topics = len(p.learning_path)
    completed = len(p.completed_topics)
    progress_pct = (completed / total_topics * 100) if total_topics > 0 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Role", p.selected_role)
    with col2:
        st.metric("Progress", f"{completed}/{total_topics} topics")
    
    # Progress bar
    st.progress(progress_pct / 100)
    
    st.divider()
    
    # Topic list with status
    st.subheader("Topics")
    
    remaining_topics = []
    for topic in p.learning_path:
        if topic in p.completed_topics:
            result = next((r for r in p.test_results if r['topic'] == topic), None)
            if result:
                status = "✅" if result['passed'] else "⚠️"
                st.success(f"{status} {topic} - {result['score']}/{result['total']}")
        else:
            remaining_topics.append(topic)
            st.info(f"○ {topic}")
    
    st.divider()
    
    # Take test section
    if remaining_topics and not st.session_state.show_test:
        st.subheader("Take a Test")
        topic = st.selectbox("Select topic", remaining_topics)
        
        if st.button("📝 Start Test", type="primary"):
            with st.spinner(f"Generating test for {topic}..."):
                try:
                    _, _, do_agent, _ = create_agents(st.session_state.opik_tracer)
                    prompt = f"Role: {p.selected_role}\nTopic: {topic}\nGenerate test."
                    response = run_agent(do_agent, prompt)
                    
                    st.session_state.current_topic = topic
                    st.session_state.current_questions = parse_questions(response, topic)
                    st.session_state.show_test = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Active test
    elif st.session_state.show_test:
        st.subheader(f"Test: {st.session_state.current_topic}")
        
        questions = st.session_state.current_questions
        
        with st.form("test_form"):
            answers = {}
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                answers[i] = st.radio(
                    f"Answer Q{i+1}",
                    options=['A', 'B', 'C', 'D'],
                    format_func=lambda x, q=q: f"{x}) {q['options'].get(x, '')}",
                    key=f"q_{i}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
            
            if st.form_submit_button("✅ Submit", type="primary"):
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
                st.session_state.show_test = False
                
                if passed:
                    st.success(f"🎉 Passed! {score}/{total}")
                else:
                    st.warning(f"Score: {score}/{total}. Keep practicing!")
                
                time.sleep(2)
                st.rerun()
        
        if st.button("← Cancel"):
            st.session_state.show_test = False
            st.rerun()
    
    # All completed
    else:
        st.success("🎉 All topics completed!")
        if st.button("🚀 Go to Next Steps", type="primary"):
            st.session_state.page = 'go'
            st.rerun()


def render_go():
    """Go page - Community and project recommendations."""
    st.header("🚀 GO: Next Steps")
    
    p = st.session_state.progress
    
    if not p.completed_topics:
        st.info("👈 Complete some tests in DO first")
        return
    
    # Summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Role", p.selected_role)
    with col2:
        st.metric("Skills", len(p.completed_topics))
    
    st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌐 Find Community", type="primary", use_container_width=True):
            with st.spinner("Finding communities..."):
                try:
                    _, _, _, go_agent = create_agents(st.session_state.opik_tracer)
                    prompt = f"Role: {p.selected_role}\nSkills: {', '.join(p.completed_topics)}\nFind communities."
                    st.session_state.community_response = run_agent(go_agent, prompt)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if st.button("📋 Get Project Ideas", type="primary", use_container_width=True):
            with st.spinner("Generating project plan..."):
                try:
                    _, _, _, go_agent = create_agents(st.session_state.opik_tracer)
                    prompt = f"Role: {p.selected_role}\nSkills: {', '.join(p.completed_topics)}\nGenerate project plan."
                    st.session_state.project_response = run_agent(go_agent, prompt)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Display responses
    if st.session_state.community_response:
        with st.expander("🌐 Community Resources", expanded=True):
            st.markdown(st.session_state.community_response)
    
    if st.session_state.project_response:
        with st.expander("📋 Project Plan", expanded=True):
            st.markdown(st.session_state.project_response)


def render_profile():
    """Profile page - Progress, badge, export."""
    st.header("👤 Your Profile")
    
    p = st.session_state.progress
    
    if not p.passion:
        st.info("Start your journey on the Home page!")
        return
    
    # Badge section
    avg_score = p.calculate_average_score()
    completion = len(p.completed_topics) / len(p.learning_path) if p.learning_path else 0
    badge_level, badge_class, badge_color = get_badge_info(p)
    
    # Badge card
    st.markdown(f"""
    <div class="badge-card {badge_class}">
        <h1 style="font-size: 3rem; margin: 0;">{badge_level}</h1>
        <h3>{p.selected_role or 'Explorer'}</h3>
        <p>Score: {avg_score:.1f}% | Topics: {len(p.completed_topics)}/{len(p.learning_path)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Topics Completed", len(p.completed_topics))
    with col2:
        st.metric("Average Score", f"{avg_score:.1f}%")
    with col3:
        st.metric("Completion", f"{completion*100:.0f}%")
    
    st.divider()
    
    # Test results
    if p.test_results:
        st.subheader("📊 Test History")
        for result in p.test_results:
            status = "✅" if result['passed'] else "❌"
            st.write(f"{status} **{result['topic']}**: {result['score']}/{result['total']}")
    
    st.divider()
    
    # Export & Share section
    st.subheader("📤 Export & Share")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export progress JSON
        json_data = p.to_json()
        st.download_button(
            "💾 Export Progress",
            data=json_data,
            file_name="synthesis_progress.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Copy shareable badge
        badge_text = generate_shareable_badge(p)
        st.download_button(
            "🏅 Share Badge",
            data=badge_text,
            file_name="synthesis_badge.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.divider()
    
    # Reset
    if st.button("🔄 Start Fresh", type="secondary", use_container_width=True):
        st.session_state.progress = UserProgress()
        st.session_state.page = 'home'
        st.session_state.community_response = ''
        st.session_state.project_response = ''
        st.rerun()


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application entry point."""
    
    # Initialize Opik tracer
    opik_key = os.getenv("OPIK_API_KEY")
    if opik_key and not st.session_state.opik_tracer:
        try:
            from opik.integrations.adk import OpikTracer
            st.session_state.opik_tracer = OpikTracer(
                name="synthesis-web",
                tags=["synthesis", "streamlit"],
                metadata={"environment": "production"},
                project_name="synthesis"
            )
        except Exception:
            pass
    
    # Render navigation
    render_navigation()
    
    # Render current page
    page = st.session_state.page
    
    if page == 'home':
        render_home()
    elif page == 'plan':
        if not st.session_state.api_key_valid:
            st.warning("Please configure your API key on the Home page first")
            render_home()
        else:
            render_plan()
    elif page == 'do':
        if not st.session_state.api_key_valid:
            st.warning("Please configure your API key on the Home page first")
            render_home()
        else:
            render_do()
    elif page == 'go':
        if not st.session_state.api_key_valid:
            st.warning("Please configure your API key on the Home page first")
            render_home()
        else:
            render_go()
    elif page == 'profile':
        render_profile()


if __name__ == "__main__":
    main()
