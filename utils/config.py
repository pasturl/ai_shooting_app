import streamlit as st


def load_api_tokens():
    try:
        REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
        ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
        return REPLICATE_API_TOKEN, ANTHROPIC_API_KEY
    except KeyError:
        st.error("""
            Missing API tokens in secrets.toml file. 
            
            Please create a .streamlit/secrets.toml file with:
            ```
            REPLICATE_API_TOKEN = "your_replicate_token_here"
            ANTHROPIC_API_KEY = "your_anthropic_key_here"
            ```
        """)
        st.stop()

# Custom CSS for better UI
def apply_custom_css():
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 20px;
        }
        .stProgress>div>div>div {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True) 