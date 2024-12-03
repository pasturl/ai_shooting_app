import streamlit as st
import sys
from pathlib import Path
import logging
from botocore.exceptions import ClientError

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the root directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from utils.config import apply_custom_css, load_api_tokens
from utils.db import DynamoDBManager

# Initialize page config and styling
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
try:
    apply_custom_css()
except Exception as e:
    logger.error(f"Failed to apply custom CSS: {str(e)}")
    st.warning("Some styling elements might not be displayed correctly")

# Initialize database connection
try:
    db = DynamoDBManager()
    
    # Fetch metrics from database
    total_images = db.get_total_images_generated()
    available_models = db.get_available_models_count()
    active_users = db.get_active_users_count()
    
except ClientError as e:
    logger.error(f"Database connection error: {str(e)}")
    st.error("Unable to connect to the database. Some features may be limited.")
    total_images = "N/A"
    available_models = "N/A"
    active_users = "N/A"
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    st.error("An unexpected error occurred while initializing the application")
    total_images = "N/A"
    available_models = "N/A"
    active_users = "N/A"

# Configure the main page
st.title("🎨 AI Image Generator")

st.markdown("""
## Welcome to the AI Image Generator!

This application allows you to:

### 🖼️ Generate Images
Create stunning AI-generated images using fine-tuned models.

### 🔧 Fine-Tune Models
Train and customize your own AI models for specific styles or purposes.

### ⚙️ Admin Panel
Manage your models and monitor training progress.

---
**Get Started:**
1. Use the sidebar to navigate between different sections
2. Start with the Generate page to create images
3. Visit the Fine-Tune page to train your own models
4. Check the Admin panel to manage your models

""")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Team")

# Add metrics with error handling
try:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images Generated", total_images)
    with col2:
        st.metric("Available Models", available_models)
    with col3:
        st.metric("Active Users", active_users)
except Exception as e:
    logger.error(f"Error displaying metrics: {str(e)}")
    st.warning("Unable to display current statistics")

# Add any common utilities or configurations here 