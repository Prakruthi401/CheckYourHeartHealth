"""Restrained shared layout for the research demo."""
import streamlit as st

def apply_global_styles():
    st.markdown("""<style>
    .stApp, [data-testid="stAppViewContainer"] {background:#101722;color:#e5eaf1;}
    [data-testid="stHeader"] {background:#101722;}
    [data-testid="stSidebar"] {background:#151f2d;border-right:1px solid #2a3545;}
    .block-container {max-width:1180px;padding:2.5rem 2.25rem 5rem;}
    h1 {font-size:2.15rem!important;font-weight:650!important;letter-spacing:-.045em;}
    h2 {font-size:1.55rem!important;letter-spacing:-.025em;}
    h3 {font-size:1.08rem!important;font-weight:600!important;}
    h1,h2,h3 {color:#edf2f8!important;}
    p, label {line-height:1.6;}
    [data-testid="stCaptionContainer"] {color:#9daec2;}
    [data-testid="stForm"] {border:1px solid #324054;border-radius:12px;padding:1.5rem;background:#151f2d;}
    [data-testid="stMetric"] {padding:1rem 1.25rem;background:#192536;border:1px solid #334258;border-radius:10px;}
    [data-testid="stMetricValue"] {font-size:2rem;}
    .stButton>button,.stFormSubmitButton>button {border:1px solid #617c91;border-radius:7px;background:#25465a;color:#f4f7fa;min-height:2.65rem;}
    .stButton>button:hover,.stFormSubmitButton>button:hover {background:#335d73;border-color:#95b8cc;color:white;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {padding:.3rem 0;}
    .nav-card {padding:.1rem 0;margin:1.25rem 0 .35rem;border-top:1px solid #344052;}
    .nav-card h3 {color:#9daec2!important;font-size:.75rem!important;letter-spacing:.12em;text-transform:uppercase;}
    .user-profile {background:transparent!important;border-bottom:1px solid #344052;border-radius:0!important;padding:1rem 0!important;}
    .user-avatar {background:#31536b!important;border-radius:10px!important;}
    .role-badge {background:#2e4155!important;color:#cfdeec!important;font-size:.72rem!important;}
    .admin-card {background:#192536;border:1px solid #334258;border-radius:10px;padding:1.25rem;}
    [data-testid="stExpander"] {border-color:#344052;}
    @media(max-width:700px){.block-container {padding:1.5rem 1rem 3rem;}h1{font-size:1.8rem!important;}}
    </style>""", unsafe_allow_html=True)

def apply_custom_styles():
    apply_global_styles()
