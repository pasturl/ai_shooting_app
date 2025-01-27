import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
import replicate
import requests
from PIL import Image
from io import BytesIO
import sys
import random
import logging
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('batch_generation.log')
    ]
)

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.config import apply_custom_css, load_api_tokens
from utils.db import DynamoDBManager
from utils.error_handling import handle_api_errors, init_database

st.set_page_config(page_title="Batch Generate Images", layout="wide")
apply_custom_css()

class BatchImageGenerator:
    def __init__(self):
        """Initialize the generator with Replicate API token."""
        try:
            REPLICATE_API_TOKEN, _ = load_api_tokens()
            self.api_token = REPLICATE_API_TOKEN
            self.client = replicate.Client(api_token=self.api_token)
            self.db = init_database()
        except Exception as e:
            st.error(f"Error initializing BatchImageGenerator: {str(e)}")
            raise

    def create_output_directory(self) -> Path:
        """Create a timestamped output directory for the generated images."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("batch_generated_images") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def download_image(self, url: str) -> Image.Image:
        """Download an image from a URL."""
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    @handle_api_errors
    def generate_image(self, prompt: str, config: dict, output_dir: Path, progress_bar) -> None:
        """Generate a single image based on the provided prompt and configuration."""
        try:
            # Combine prompt with configuration
            api_params = config.copy()
            api_params["prompt"] = prompt

            # Generate random seed if needed
            if api_params.get("seed", -1) == -1:
                api_params["seed"] = random.randint(1, 1000000000)

            st.info(f"Generating image for prompt: {prompt[:100]}...")
            
            # Run the model
            output = self.client.run(
                f"{config['model_path']}:{config['model_version']}",
                input=api_params
            )

            # Process and save the generated images
            if isinstance(output, list):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c.isspace())
                
                for idx, img_url in enumerate(output):
                    # Download and save image
                    image = self.download_image(img_url)
                    filename = f"{timestamp}_{clean_prompt}_{idx + 1}.png"
                    image_path = output_dir / filename
                    image.save(image_path, "PNG")

                    # Save parameters
                    params_path = output_dir / f"{timestamp}_{clean_prompt}_{idx + 1}_params.json"
                    config_with_seed = config.copy()
                    config_with_seed["seed"] = api_params["seed"]
                    full_params = {
                        "prompt": prompt,
                        **config_with_seed
                    }
                    with open(params_path, 'w') as f:
                        json.dump(full_params, f, indent=4)

                    # Update progress
                    progress_bar.progress((idx + 1) / len(output))

        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            logging.error(f"Error generating image: {str(e)}")

    def process_batch(self, prompts: list, config: dict, status_area) -> None:
        """Process multiple prompts using the provided configuration."""
        try:
            # Verify config has necessary parameters
            required_params = [
                "width", "height", "model", "model_path", "model_version",
                "lora_scale", "guidance_scale", "prompt_strength",
                "num_inference_steps", "num_outputs", "seed"
            ]
            for param in required_params:
                if param not in config:
                    raise ValueError(f"Missing required parameter '{param}' in config")

            # Create output directory
            output_dir = self.create_output_directory()
            st.info(f"Output directory created: {output_dir}")
            
            # Save configuration and prompts
            with open(output_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=4)
            with open(output_dir / "prompts.txt", 'w') as f:
                f.write("\n".join(prompts))

            # Process each prompt
            total_prompts = len(prompts)
            for idx, prompt in enumerate(prompts, 1):
                status_area.info(f"Processing prompt {idx}/{total_prompts}")
                progress_bar = st.progress(0)
                self.generate_image(prompt, config, output_dir, progress_bar)
                progress_bar.empty()

            st.success(f"Batch processing complete! Output directory: {output_dir}")

        except Exception as e:
            st.error(f"Error during batch processing: {str(e)}")
            logging.error(f"Error during batch processing: {str(e)}")

def main():
    st.title("🎯 Batch Image Generation")
    
    try:
        generator = BatchImageGenerator()
        
        # File upload section
        st.header("Upload Files")
        col1, col2 = st.columns(2)
        
        with col1:
            prompts_file = st.file_uploader(
                "Upload Prompts File (TXT)",
                type=['txt'],
                help="Text file with one prompt per line"
            )

        with col2:
            config_file = st.file_uploader(
                "Upload Configuration File (JSON)",
                type=['json'],
                help="JSON file containing generation parameters"
            )

        # Display file contents if uploaded
        if prompts_file and config_file:
            st.header("File Contents")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prompts")
                prompts_content = prompts_file.read().decode()
                prompts = [line.strip() for line in prompts_content.split('\n') if line.strip()]
                st.text_area("Prompts Preview", prompts_content, height=200)
                st.info(f"Total prompts: {len(prompts)}")

            with col2:
                st.subheader("Configuration")
                config_content = config_file.read().decode()
                try:
                    config = json.loads(config_content)
                    st.json(config)
                except json.JSONDecodeError:
                    st.error("Invalid JSON configuration file")
                    return

            # Generation section
            st.header("Generate Images")
            if st.button("Start Batch Generation", use_container_width=True):
                status_area = st.empty()
                generator.process_batch(prompts, config, status_area)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 