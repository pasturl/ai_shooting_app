# Move all your current generation code here
import streamlit as st
import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('generation.log')
    ]
)

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Now we can import from utils
from utils.config import apply_custom_css, load_api_tokens
from utils.db import DynamoDBManager
from utils.error_handling import handle_api_errors, init_database
from utils.training import update_all_training_status, check_training_status

st.set_page_config(page_title="Generate Images", page_icon="🖼️", layout="wide")

# Rest of your imports
import replicate
from PIL import Image
import requests
from io import BytesIO
import time
import zipfile
import base64
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
import json
import asyncio
from datetime import datetime
import pathlib
import re
import concurrent.futures
from functools import partial
from typing import List, Dict
import random

# Initialize page config and styling
apply_custom_css()
# Load API tokens
REPLICATE_API_TOKEN, ANTHROPIC_API_KEY = load_api_tokens()

# Initialize database with error handling
db = init_database()

# Initialize Replicate client
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

class FluxImageGenerator:
    def __init__(self):
        # Update all model statuses first
        update_all_training_status(db)
        
        # Get available models from database
        available_models = db.get_all_models()
        
        # Convert to the format needed by the generator
        self.MODELS = {
            f"{model['name']} ({model['owner']})": {
                "path": model['model_path'],
                "version": model['version_id'],
                "trigger_word": model.get('training_params', {}).get('trigger_word', ''),
                "status": model['status']
            } for model in available_models if model['status'] == 'succeeded'
        }
        
        if not self.MODELS:
            st.warning("No trained models available in the database. Please train a model first.")
            self.current_model = None
        else:
            # Default to first model
            self.current_model = list(self.MODELS.keys())[0]
    
    def set_model(self, model_name):
        """Set the current model"""
        if model_name in self.MODELS:
            self.current_model = model_name
            return True
        return False
    
    def get_current_model_info(self):
        """Get current model information"""
        if self.current_model and self.current_model in self.MODELS:
            return self.MODELS[self.current_model]
        return None
    
    def get_trigger_word(self):
        """Get trigger word for current model"""
        model_info = self.get_current_model_info()
        return model_info['trigger_word'] if model_info else None
        
    def download_image(self, url):
        """Download image from URL and return PIL Image object"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            st.error(f"Error downloading image: {str(e)}")
            return None
        
    @handle_api_errors
    def generate_image(self, prompt, params):
        """Generate a single image"""
        try:
            model_info = self.MODELS[self.current_model]
            logging.info(f"Generating image using model: {self.current_model}")
            logging.info(f"Model path: {model_info['path']}")
            logging.info(f"Model version: {model_info['version']}")
            
            output = client.run(
                f"{model_info['path']}:{model_info['version']}",
                input={
                    "prompt": prompt,
                    **params
                },
                api_token=REPLICATE_API_TOKEN
            )
            
            
            if output and isinstance(output, list) and len(output) > 0:
                image = self.download_image(output[0])
                if image:
                    # Create timestamp for unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Create base directory if it doesn't exist
                    base_dir = pathlib.Path("generated_images")
                    base_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create safe filename from prompt
                    safe_prompt = re.sub(r'[^\w\s-]', '', prompt)[:30]
                    safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt).strip('-_')
                    
                    # Save image with timestamp and prompt in filename
                    filename = f"{timestamp}_{safe_prompt}.{params['output_format']}"
                    save_path = base_dir / filename
                    
                    # Save the image with the specified format and quality
                    image.save(
                        save_path, 
                        format=params['output_format'].upper(),
                        quality=params['output_quality']
                    )
                    
                    st.success(f"Image saved to: {save_path}")
                    return image
                
            return None
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None

    def generate_images_parallel(self, variations, params, max_workers=3):
        """Generate multiple images in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function with fixed params
            generate_func = partial(self.generate_image, params=params)
            
            # Submit all tasks and get future objects
            future_to_prompt = {
                executor.submit(generate_func, prompt): (i, prompt) 
                for i, prompt in enumerate(variations)
            }
            
            # Dictionary to store results
            results = {}
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_prompt):
                idx, prompt = future_to_prompt[future]
                try:
                    image = future.result()
                    results[idx] = {
                        'image': image,
                        'prompt': prompt,
                        'success': image is not None
                    }
                except Exception as e:
                    results[idx] = {
                        'image': None,
                        'prompt': prompt,
                        'success': False,
                        'error': str(e)
                    }
            
            # Sort results by original index
            return [results[i] for i in range(len(variations))]

def create_mock_variations(original_prompt: str, selected_model: str) -> Dict:
    """Generate mock variations for debug mode"""
    mock_variations = []
    styles = ["cinematic", "street photography", "fashion editorial", "urban lifestyle"]
    focuses = ["product detail", "full body shot", "environmental context", "artistic composition"]
    
    for i in range(10):
        mock_variations.append({
            "prompt": f"Mock variation {i+1} for: {original_prompt}",
            "style": random.choice(styles),
            "focus": random.choice(focuses)
        })
    
    return {"variations": mock_variations}

def create_mock_image() -> Image.Image:
    """Create a mock image for debug mode"""
    width, height = 512, 512
    mock_image = Image.new('RGB', (width, height))
    pixels = mock_image.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (random.randint(0, 255), 
                          random.randint(0, 255), 
                          random.randint(0, 255))
    return mock_image

def generate_prompt_variations(original_prompt, selected_model):
    """Generate 10 variations of the input prompt using Claude"""
    llm = ChatAnthropic(
        anthropic_api_key=ANTHROPIC_API_KEY,
        model="claude-3-sonnet-20240229",
        temperature=0.8,
        max_tokens=2000
    )
    
    template = """You are an expert AI assistant specializing in generating professional-grade fashion photography prompts, with deep knowledge of commercial product photography, lighting techniques, and contemporary fashion campaigns.

CORE OBJECTIVES:
1. Generate 10 highly detailed, commercial-quality variations of the provided prompt
2. Ensure each prompt is optimized for AI image generation
3. Maintain consistent focus on the footwear product while creating compelling scene compositions
4. ALWAYS focus on the footwear product as the hero of the image.
5. NEVER include people on the images.

PROMPT REQUIREMENTS:
Each variation MUST include:

1. TECHNICAL SPECIFICATIONS
- Precise camera angles (e.g., "shot at 35mm, f/2.8 aperture")
- Specific lighting setup (e.g., "three-point lighting with rim light")
- Resolution and quality markers (e.g., "8K, hyperrealistic, photorealistic")
- Post-processing style (e.g., "slight film grain, Kodak Portra 400 colors")

2. ENVIRONMENT & COMPOSITION
- Detailed setting description
- Time of day and weather conditions
- Generate realistic scenes that match a footwear campaign.
- Specific composition rules (Rule of thirds, leading lines, etc.)
- Distance and framing (close-up, medium shot, wide shot)

4. PRODUCT EMPHASIS
- {selected_model} placement and interaction with environment
- Key product features to highlight
- Natural integration into the scene

5. ATMOSPHERE & MOOD
- Color palette and color grading
- Atmospheric elements (fog, shadows, reflections)
- Emotional tone of the image

FORBIDDEN ELEMENTS:
- Avoid generic descriptors (beautiful, nice, amazing)
- No unrealistic or physically impossible scenarios
- Avoid overshadowing the product with complex scenes
- No technical impossibilities for AI generation
    
    EXAMPLE PROMPT STRUCTURE:
    "Professional fashion campaign shot of {selected_model} sneakers, captured at 35mm with f/2.8 aperture. Three-point lighting setup with key light at 45 degrees. Setting: Modern concrete architecture with strong geometric shadows, shot during golden hour. Product positioned at lower third, emphasized by natural leading lines in architecture. Kodak Portra 400 color grading, slight film grain, 8K resolution. Style: minimal editorial fashion photography with strong architectural elements."

    Remember to make each variation unique while maintaining consistent professional quality and commercial viability. Focus on creating prompts that combine technical precision with artistic vision, always keeping the product as the hero of the image. 
    product: {selected_model}
    Original prompt: {original_prompt}
    
    Return ONLY a JSON response in this exact format:
    {{
        "variations": [
            {{
                "prompt": "complete prompt text",
                "style": "artistic style used",
                "focus": "main focus/perspective of this variation"
            }},
            // ... (9 more variations)
        ]
    }}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    try:
        with st.spinner("🤔 Generating prompt variations..."):
            response = llm.invoke(prompt.format(original_prompt=original_prompt, selected_model=selected_model))
            
            try:
                json_str = response.content.strip()
                response_data = json.loads(json_str)
                
                variation_container = st.container()
                
                with variation_container:
                    st.subheader("📝 Generated Prompt Variations")
                    cols = st.columns(2)
                    
                    for idx, var_data in enumerate(response_data.get("variations", []), 1):
                        col = cols[0] if idx <= 5 else cols[1]
                        with col:
                            with st.expander(f"Variation {idx}", expanded=False):
                                st.markdown(f"""
                                **Prompt:** {var_data['prompt']}
                                
                                **Style:** {var_data['style']}
                                
                                **Focus:** {var_data['focus']}
                                """)
                
                variations = [item["prompt"] for item in response_data.get("variations", [])]
                st.session_state.variation_data = response_data.get("variations", [])
                return variations
                
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON response: {str(e)}")
                return [original_prompt]

    except Exception as e:
        st.error(f"Error generating prompt variations: {str(e)}")
        st.exception(e)
        return [original_prompt]

def create_safe_filename(prompt, max_length=30):
    """Create a safe filename from the prompt"""
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt)
    safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt).strip('-_')
    return safe_prompt[:max_length]

def save_generated_image(image, prompt, variation_num, params):
    """Save the generated image and its parameters in a timestamped folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = create_safe_filename(prompt)
    base_dir = pathlib.Path("generated_images")
    folder_name = f"{timestamp}_{safe_prompt}"
    save_dir = base_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    image_filename = f"variation_{variation_num}.png"
    image_path = save_dir / image_filename
    image.save(image_path, "PNG")
    
    params_filename = f"variation_{variation_num}_params.txt"
    params_path = save_dir / params_filename
    with open(params_path, "w") as f:
        f.write(f"Original Prompt: {prompt}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Variation: {variation_num}\n")
        f.write("\nParameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    return save_dir

# Initialize the image generator
generator = FluxImageGenerator()

# Sidebar controls
st.sidebar.title("Generation Settings")

# Add refresh button in sidebar
if st.sidebar.button("🔄 Refresh Models", use_container_width=True):
    with st.spinner("Checking model status..."):
        updates = update_all_training_status(db)
        if updates:
            st.sidebar.success(f"Updated {len(updates)} models")
            # Reinitialize generator to get updated models
            generator = FluxImageGenerator()
        else:
            st.sidebar.info("No status changes detected")

if generator.MODELS:
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(generator.MODELS.keys()),
        index=list(generator.MODELS.keys()).index(generator.current_model) if generator.current_model else 0
    )
    generator.set_model(selected_model)
    logging.info(f"Selected model changed to: {selected_model}")
    
    # Show model information
    with st.sidebar.expander("Model Information", expanded=False):
        model_info = generator.get_current_model_info()
        st.markdown(f"""
        - **Model Path:** `{model_info['path']}`
        - **Trigger Word:** `{model_info['trigger_word']}`
        - **Version:** `{model_info['version'][:8]}...`
        - **Status:** `{model_info['status']}`
        """)
    
    # Image format selection
    format_options = {
        "Square (1:1)": {"width": 1024, "height": 1024},
        "Portrait (3:4)": {"width": 768, "height": 1024},
        "Landscape (4:3)": {"width": 1024, "height": 768},
        "Widescreen (16:9)": {"width": 1024, "height": 576},
    }

    selected_format = st.sidebar.selectbox(
        "Image Format",
        list(format_options.keys())
    )

    # Update params dictionary
    params = {
        "num_outputs": st.sidebar.selectbox("Number of Outputs", [1, 2, 3, 4], index=0),
        "width": format_options[selected_format]["width"],
        "height": format_options[selected_format]["height"],
        "model": st.sidebar.selectbox("Model", ["schnell", "dev"], index=0),
        "lora_scale": st.sidebar.slider("LoRA Scale", 0.0, 2.0, 1.0, 0.1),        
        "output_format": st.sidebar.selectbox("Output Format", ["png", "jpg", "webp"], index=0),
        "guidance_scale": st.sidebar.slider("Guidance Scale", 1.0, 20.0, 3.5, 0.5),
        "output_quality": st.sidebar.slider("Output Quality", 1, 100, 90, 1),
        "prompt_strength": st.sidebar.slider("Prompt Strength", 0.0, 1.0, 0.8, 0.1),
        "extra_lora_scale": st.sidebar.slider("Extra LoRA Scale", 0.0, 2.0, 1.0, 0.1),
        "num_inference_steps": st.sidebar.slider("Inference Steps", 1, 50, 4, 1)
    }

    # Main area
    st.title("Generate Images")
    
    # Get trigger word and show it in the prompt area
    trigger_word = generator.get_trigger_word()
    st.markdown(f"### Using trigger word: `{trigger_word}`")
    st.markdown("Include this trigger word in your prompt to activate the model's style.")
    
    prompt = st.text_area(
        "Enter your prompt", 
        height=100,
        help=f"Describe the image you want to generate. Include the trigger word '{trigger_word}' to activate the model's style."
    )

    # Generation button
    if st.button("Generate Image"):
        if not prompt:
            st.warning("Please enter a prompt first.")
        else:
            # Ensure trigger word is in prompt
            if trigger_word and trigger_word not in prompt:
                modified_prompt = f"{trigger_word} {prompt}"
                st.info(f"Added trigger word to prompt: {modified_prompt}")
                prompt = modified_prompt
                
            with st.spinner("Generating image..."):
                try:
                    # Generate the image
                    image = generator.generate_image(prompt, params)
                    
                    if image:
                        # Display the generated image
                        st.image(image, caption=prompt, use_column_width=True)
                        
                        # Save the image
                        save_dir = save_generated_image(image, prompt, 1, params)
                        st.success(f"Image saved in: {save_dir}")
                    else:
                        st.error("Failed to generate image. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")
                    st.exception(e)
else:
    st.warning("No trained models available. Please go to the Fine-Tune section to train a model first.")