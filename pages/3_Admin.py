import streamlit as st
st.set_page_config(page_title="Admin Panel", page_icon="⚙️", layout="wide")

import replicate
import sys
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
from botocore.exceptions import ClientError
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the root directory to Python path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from utils.config import apply_custom_css, load_api_tokens
from utils.db import DynamoDBManager
from utils.training import update_all_training_status

# Initialize page config and styling
apply_custom_css()

# Load API tokens
REPLICATE_API_TOKEN, _ = load_api_tokens()

# Initialize DynamoDB manager with error handling
try:
    db = DynamoDBManager()
except Exception as e:
    logger.error(f"Failed to initialize DynamoDB manager: {str(e)}")
    st.error("Failed to connect to database. Please check your AWS credentials and network connection.")
    db = None

def get_models_dataframe():
    """Convert models data to a pandas DataFrame for display"""
    try:
        if db is None:
            st.error("Database connection is not available")
            return None
            
        models = db.get_all_models()
        
        if not models:
            return None
            
        # Extract relevant fields
        data = []
        for model in models:
            training_params = model.get('training_params', {})
            created_at = datetime.fromisoformat(model.get('created_at', '')) if model.get('created_at') else datetime.min
            data.append({
                'Model ID': model['model_id'],
                'Name': model['name'],
                'Owner': model['owner'],
                'Status': model['status'],
                'Created': created_at,
                'Updated': model.get('updated_at', 'N/A'),
                'Trigger Word': training_params.get('trigger_word', 'N/A'),
                'Steps': training_params.get('steps', 'N/A'),
                'Training ID': training_params.get('training_id', 'N/A'),
                'Version': model.get('version_id', 'N/A')[:8] if model.get('version_id') else 'N/A'
            })
        
        # Create DataFrame and sort by Created date
        df = pd.DataFrame(data)
        df = df.sort_values('Created', ascending=False)
        
        # Format the Created column back to string for display
        df['Created'] = df['Created'].dt.strftime('%Y-%m-%d %H:%M')
        
        return df
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"DynamoDB error: {error_code} - {error_message}")
        st.error(f"Database error: {error_message}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.error("An unexpected error occurred while fetching data")
        return None

# Main UI
st.title("Admin Dashboard")

# Refresh button with last update time
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔄 Refresh Status", use_container_width=True):
        with st.spinner("Checking training status..."):
            updates = update_all_training_status(db)
            if updates:
                st.success(f"Updated {len(updates)} models:")
                for update in updates:
                    version_info = f" (Version: {update['version_id']})" if update['version_id'] != 'N/A' else ""
                    st.info(f"{update['model']}: {update['old_status']} → {update['new_status']}{version_info}")
            else:
                st.info("No status changes detected.")

with col2:
    st.markdown(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Models table
st.header("Models Overview")
df = get_models_dataframe()

if df is not None:
    # Add styling
    def color_status(val):
        colors = {
            'created': 'background-color: #ffd700',  # Gold
            'training': 'background-color: #87ceeb',  # Sky Blue
            'succeeded': 'background-color: #90ee90',  # Light Green
            'failed': 'background-color: #ffcccb',    # Light Red
            'canceled': 'background-color: #d3d3d3'   # Light Gray
        }
        return colors.get(val, '')

    # Apply styling and display
    styled_df = df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Export options
    if st.button("Export to CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"models_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
else:
    st.info("No models found in the database") 