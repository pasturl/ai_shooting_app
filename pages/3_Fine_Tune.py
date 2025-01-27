"""
Fine-Tuning Page for Flux Image Generator

This module provides a user interface for fine-tuning custom LoRA models using the Replicate API.
It includes comprehensive controls for dataset upload, training parameters, and model configuration.

Author: Your Name
Date: 2024
"""


import streamlit as st
st.set_page_config(page_title="Fine Tune", page_icon="F", layout="wide")

import replicate
import os
from PIL import Image
import zipfile
import tempfile
from datetime import datetime
import sys
from pathlib import Path
import json
import time
from decimal import Decimal

# Add the root directory to Python path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from utils.config import apply_custom_css, load_api_tokens
from utils.db import DynamoDBManager
from utils.training import update_all_training_status


def display_fine_tuning_tips():
    """Display comprehensive tips for successful LoRA fine-tuning."""
    st.sidebar.markdown("""
    # 📚 Fine-Tuning Tips
    
    ## 📊 Dataset Requirements
    - **Size**: 12-18 high-quality images
    - **Resolution**: 1024x1024 or larger
    - **Format**: File names must be captions (e.g., a_photo_of_TOK.png)
    
    ## 🎯 Image Selection
    ### For Style LoRAs
    - Select images highlighting distinctive style features
    - Use varied subjects with consistent style
    - Avoid dataset domination by specific elements
    
    ### For Character LoRAs
    - Include different settings and expressions
    - Maintain consistent appearance (hair, age)
    - Limit hand-framing poses to prevent hallucinations
    
    ## ⚙️ Training Parameters
    - Use generic proper names for trigger words
    - LoRA Rank: 16-32 (up to 64 for likeness)
    - Increase steps for weaker datasets
    - Dataset quality affects style flexibility
    
    ## 🎨 Inference Tips
    - Pair trigger words with gender terms
    - Adjust LoRA strength (0.8-0.95) for style control
    
    ## 📝 Example Fine-Tunes
    - [Watercolor Style](https://replicate.com/lucataco/flux-watercolor)
    - [Pixar Cars Style](https://replicate.com/fofr/flux-pixar-cars)
    - [Vintage 2004](https://replicate.com/fofr/flux-2004)
    """)

# Initialize styling and load tokens
apply_custom_css()
REPLICATE_API_TOKEN, _ = load_api_tokens()

# Initialize DynamoDB manager
db = DynamoDBManager()
db.create_table_if_not_exists()

def validate_zip_contents(zip_file):
    """Validate the contents of the uploaded ZIP file"""
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            if not file_list:
                return False, "ZIP file is empty"
            
            valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
            image_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in valid_extensions)]
            
            if not image_files:
                return False, "No valid image files found in ZIP"
            
            if len(image_files) < 10:
                return False, f"Not enough images. Found {len(image_files)}, minimum required is 10"
            
            total_size = sum(zip_ref.getinfo(file).file_size for file in image_files)
            max_size_mb = 500  # Changed from 1000MB to 500MB
            if total_size > max_size_mb * 1024 * 1024:
                return False, f"Total size of images exceeds {max_size_mb}MB"
            
            return True, f"Found {len(image_files)} valid images"
            
    except zipfile.BadZipFile:
        return False, "Invalid ZIP file"
    except Exception as e:
        return False, f"Error validating ZIP: {str(e)}"

def create_training_model(owner, model_name, visibility, description, trigger_word, advanced_options):
    """Create a new model on Replicate"""
    try:
        model = replicate.models.create(
            owner=owner,
            name=model_name,
            visibility=visibility,
            hardware="gpu-t4",
            description=description
        )
        
        # Save model information to DynamoDB
        model_data = {
            'owner': owner,
            'name': model_name,
            'visibility': visibility,
            'description': description,
            'replicate_id': model.id,
            'status': 'created',     
            'model_path': f"{owner}/{model_name}",
            'trigger_word': trigger_word,
            'advanced_parameters': advanced_options
        }
        
        if db.save_model(model_data):
            return model
        else:
            st.error("Failed to save model information to database")
            return None
            
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None

def convert_floats_to_decimal(obj):
    """Recursively convert float values to Decimal for DynamoDB compatibility"""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(v) for v in obj]
    return obj

def start_training(model, training_file, steps, trigger_word, hf_token=None, hf_repo_id=None, **advanced_options):
    """Start the training process"""
    if not trigger_word:
        raise ValueError("trigger_word is required for training")
    
    # Ensure trigger word is properly formatted
    trigger_word = str(trigger_word).strip().upper()
    
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(training_file.getvalue())
            tmp_file_path = tmp_file.name

        # Prepare the training inputs with the file opened in binary mode
        training_inputs = {
            "input_images": open(tmp_file_path, "rb"),
            "steps": int(steps),
            "lora_rank": int(advanced_options.get("lora_rank", 16)),
            "optimizer": advanced_options.get("optimizer", "adamw8bit"),
            "batch_size": int(advanced_options.get("batch_size", 1)),
            "resolution": advanced_options.get("resolution", "512,768,1024"),
            "autocaption": bool(advanced_options.get("autocaption", True)),
            "trigger_word": trigger_word,
            "learning_rate": float(advanced_options.get("learning_rate", 0.0004)),
            "wandb_project": "flux_train_replicate",
            "wandb_save_interval": 100,
            "caption_dropout_rate": float(advanced_options.get("caption_dropout_rate", 0.05)),
            "cache_latents_to_disk": bool(advanced_options.get("cache_latents_to_disk", False)),
            "wandb_sample_interval": 100
        }

        # Add optional HuggingFace parameters if provided
        if hf_token and hf_repo_id:
            training_inputs["hf_token"] = hf_token
            training_inputs["hf_repo_id"] = hf_repo_id

        # Optionally, print training inputs for debugging (excluding the file object)
        debug_inputs = {k: v for k, v in training_inputs.items() if k != "input_images"}
        st.write("Training Inputs:", debug_inputs)

        # Create the training
        training = replicate.trainings.create(
            destination=f"{model.owner}/{model.name}",
            version="ostris/flux-dev-lora-trainer:e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497",
            input=training_inputs
        )

        # Update model status and training parameters in database
        model_id = f"{model.owner}/{model.name}"
        training_params = {
            'steps': steps,
            'trigger_word': trigger_word,
            'hf_integration': bool(hf_token and hf_repo_id),
            'training_id': training.id,
            'started_at': datetime.now().isoformat(),
            'trainer_version': "ostris/flux-dev-lora-trainer:e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497",
            'advanced_options': convert_floats_to_decimal(advanced_options)  # Convert floats in advanced options
        }
        
        # Convert any remaining float values in training_params
        training_params = convert_floats_to_decimal(training_params)
        
        # Add a new method to update training parameters
        update_expr = """
            SET #status = :status, 
                training_params = :training_params,
                updated_at = :updated_at
        """
        
        expr_names = {"#status": "status"}
        expr_values = {
            ":status": "training",
            ":training_params": training_params,
            ":updated_at": datetime.now().isoformat()
        }
        
        try:
            db.table.update_item(
                Key={'model_id': model_id},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values
            )
        except Exception as db_error:
            st.error(f"Error updating training parameters: {str(db_error)}")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return training
    except Exception as e:
        st.error(f"Error starting training: {str(e)}")
        return None

def update_training_status(training_id=None):
    """Update the status of a training run"""
    try:
        if not training_id:
            # Get all models in training status
            models = db.get_all_models()
            training_models = [m for m in models if m.get('status') == 'training']
            
            for model in training_models:
                training_params = model.get('training_params', {})
                if training_id := training_params.get('training_id'):
                    try:
                        # Get training status from Replicate
                        training = replicate.trainings.get(training_id)
                        
                        # Only update version_id if training is completed and has a version
                        if training.status == 'succeeded' and training.version:
                            # Extract only the ID part after the colon
                            version_id = training.version.split(':')[-1] if ':' in training.version else training.version
                        else:
                            version_id = None
                        
                        # Update status in database
                        db.update_model_training_status(
                            model_id=model['model_id'],
                            training_status=training.status,
                            version_id=version_id  # Now contains only the ID portion
                        )
                        
                        # Display current status
                        st.sidebar.info(f"Model: {model['name']}\nStatus: {training.status}")
                        
                        if training.status == 'succeeded':
                            st.sidebar.success(f"Training completed for {model['name']}!")
                        elif training.status in ['failed', 'canceled']:
                            st.sidebar.error(f"Training {training.status} for {model['name']}")
                    except Exception as e:
                        st.sidebar.error(f"Error updating status for {model['name']}: {str(e)}")
        else:
            # Update specific training
            try:
                training = replicate.trainings.get(training_id)
                return training.status
            except Exception as e:
                st.error(f"Error getting training status: {str(e)}")
                return None
                
    except Exception as e:
        st.error(f"Error updating training status: {str(e)}")
        return None

def handle_training(owner, model_name, trigger_word, uploaded_file, steps, learning_rate, 
                   batch_size, resolution, lora_rank, caption_dropout_rate, optimizer, 
                   cache_latents, autocaption, autocaption_prefix, autocaption_suffix, 
                   layers_regex, hf_token, hf_repo_id):
    """Handle the model creation and training process"""
    # Validate and format trigger word
    if not trigger_word or not trigger_word.strip():
        st.error("Trigger word is required and cannot be empty")
        return
    
    trigger_word = trigger_word.strip().upper()
    
    # Collect advanced options with proper type conversion
    advanced_options = {
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "resolution": resolution,
        "lora_rank": int(lora_rank),
        "caption_dropout_rate": float(caption_dropout_rate),
        "optimizer": optimizer,
        "cache_latents_to_disk": bool(cache_latents),
        "autocaption": bool(autocaption)
    }
    
    if autocaption:
        if autocaption_prefix:
            advanced_options["autocaption_prefix"] = autocaption_prefix
        if autocaption_suffix:
            advanced_options["autocaption_suffix"] = autocaption_suffix
    
    if layers_regex:
        advanced_options["layers_to_optimize_regex"] = layers_regex
    
    # Create the model first
    model = create_training_model(
        owner=owner,
        model_name=model_name,
        visibility="private",
        description=f"Fine-tuned model with trigger word: {trigger_word}",
        trigger_word=trigger_word,
        advanced_options=advanced_options
    )
    
    if not model:
        st.error("Failed to create model")
        return
        
    # Start the training process
    with st.spinner("Starting training..."):
        
        training = start_training(
            model=model,
            training_file=uploaded_file,
            steps=steps,
            trigger_word=trigger_word,  # Ensure trigger_word is passed
            hf_token=hf_token if hf_token else None,
            hf_repo_id=hf_repo_id if hf_repo_id else None,
            **advanced_options
        )
        
        if training:
            st.success("Training started successfully!")
            st.markdown(f"""
            ### Training Details
            - Status: {training.status}
            - Training URL: https://replicate.com/p/{training.id}
            
            You can monitor the training progress in the sidebar or at the URL above.
            Training typically takes 20-30 minutes for 1000 steps.
            """)
            
            # Start automatic status checking
            with st.spinner("Checking initial status..."):
                time.sleep(5)  # Wait a bit for training to initialize
                initial_status = update_training_status(training.id)
                if initial_status:
                    st.info(f"Initial status: {initial_status}")
            
            st.info("Use the 'Refresh Status' button in the sidebar to check progress.")

# Display existing models
st.sidebar.title("Existing Models")

# Add refresh button
if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
    with st.spinner("Checking model status..."):
        updates = update_all_training_status(db)
        if updates:
            st.sidebar.success(f"Updated {len(updates)} models")
            for update in updates:
                version_info = f" (Version: {update['version_id']})" if update['version_id'] != 'N/A' else ""
                st.sidebar.info(f"{update['model']}: {update['old_status']} → {update['new_status']}{version_info}")
        else:
            st.sidebar.info("No status changes detected")

existing_models = db.get_all_models()

if existing_models:
    st.sidebar.markdown("### Recent Models (Last 10)")
    
    # Sort models by creation date (newest first) and take last 10
    sorted_models = sorted(existing_models, 
                         key=lambda x: x.get('created_at', ''), 
                         reverse=True)[:10]
    
    for model in sorted_models:
        with st.sidebar.expander(f"{model['name']} ({model['status']})"):
            trigger_word = model.get('training_params', {}).get('trigger_word', 'N/A')
            st.markdown(f"""
            - **Owner:** {model['owner']}
            - **Status:** {model['status']}
            - **Created:** {datetime.fromisoformat(model['created_at']).strftime('%Y-%m-%d %H:%M')}
            - **Trigger Word:** {trigger_word}
            """)
            
            # Add a link to the model on Replicate
            st.markdown(f"[View on Replicate](https://replicate.com/{model['owner']}/{model['name']})")
else:
    st.sidebar.info("No models found in the database")

# Main UI
st.title("Fine-Tune Model")

# Basic Configuration
col1, col2, col3 = st.columns(3)

with col1:
    owner = st.text_input(
        "Owner Username", 
        value="pasturl",
        disabled=False,
        help="Your Replicate username"
    )

with col2:
    model_name = st.text_input(
        "Model Name",
        value=f"flux-lora-{datetime.now().strftime('%Y%m%d')}",
        help="Name for your fine-tuned model"
    )

with col3:
    trigger_word = st.text_input(
        "Trigger Word *",
        value="FLUXLORA",
        help="A unique string like FLUXLORA that will be associated with your model (REQUIRED)"
    )
    # Validate trigger word format
    if trigger_word:
        if not trigger_word.isupper() or ' ' in trigger_word:
            st.warning("Trigger word should be in UPPERCASE and contain no spaces")

# Training Steps and Captioning
st.header("Training Configuration")
col1, col2, col3, col4 = st.columns(4)

with col1:
    steps = st.number_input(
        "Training Steps",
        min_value=3,
        max_value=6000,
        value=1000,
        help="Number of training steps. Recommended range 500-4000"
    )
    estimated_cost = (steps / 1000) * 2  # $2 per 1000 steps
    st.info(f"Cost: ${estimated_cost:.2f}")

with col2:
    autocaption = st.checkbox(
        "Auto Caption Images",
        value=True,
        help="Automatically caption images using Llava v1.5 13B"
    )

with col3:
    autocaption_prefix = st.text_area(
        "Caption Prefix",
        help="Text to appear at the beginning of all generated captions. You can include your trigger word.",
        placeholder=f"Example: a photo of {trigger_word}, ",
        height=100
    )

with col4:
    autocaption_suffix = st.text_area(
        "Caption Suffix",
        help="Text to appear at the end of all generated captions. You can include your trigger word.",
        placeholder=f"Example: in the style of {trigger_word}",
        height=100
    )

# Advanced Options in a collapsible section
with st.expander("Advanced Options", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-3,
            value=4e-4,
            format="%.6f",
            help="Learning rate for training"
        )
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=4,
            value=1,
            help="Number of images to process at once"
        )
        
        resolution = st.text_input(
            "Resolution",
            value="512,768,1024",
            help="Image resolutions for training (comma-separated)"
        )
        
        lora_rank = st.number_input(
            "LoRA Rank",
            min_value=1,
            max_value=128,
            value=32,
            help="Higher ranks capture more complex features"
        )
        
    with col2:
        caption_dropout_rate = st.slider(
            "Caption Dropout Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            help="Rate at which captions are ignored during training"
        )
        
        optimizer = st.selectbox(
            "Optimizer",
            ["prodigy", "adam8bit", "adamw8bit", "lion8bit", "adam", "adamw", "lion", "adagrad", "adafactor"],
            index=2,
            help="Optimizer to use for training"
        )
        
        cache_latents = st.checkbox(
            "Cache Latents to Disk",
            value=False,
            help="Use for large datasets to prevent memory issues"
        )
        
        layers_regex = st.text_input(
            "Layers to Optimize (Regex)",
            help="Regular expression to match specific layers to optimize"
        )

# Hugging Face Integration
with st.expander("Hugging Face Integration", expanded=False):
    hf_token = st.text_input("Hugging Face Token", type="password")
    hf_repo_id = st.text_input("Hugging Face Repository ID")

# Training Data Upload
uploaded_file = st.file_uploader(
    "Upload Training Images (ZIP)",
    type=['zip'],
    help="ZIP file containing your training images (min 10 images)",
    accept_multiple_files=False
)

# Start Training Button
if st.button("Start Training", type="primary"):
    if not trigger_word or not trigger_word.strip():
        st.error("Trigger word is required")
    elif not all([owner, model_name, uploaded_file]):
        st.error("Please fill in all required fields and upload a ZIP file.")
    else:
        handle_training(
            owner=owner,
            model_name=model_name,
            trigger_word=trigger_word,
            uploaded_file=uploaded_file,
            steps=steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            resolution=resolution,
            lora_rank=lora_rank,
            caption_dropout_rate=caption_dropout_rate,
            optimizer=optimizer,
            cache_latents=cache_latents,
            autocaption=autocaption,
            autocaption_prefix=autocaption_prefix,
            autocaption_suffix=autocaption_suffix,
            layers_regex=layers_regex,
            hf_token=hf_token,
            hf_repo_id=hf_repo_id
        )