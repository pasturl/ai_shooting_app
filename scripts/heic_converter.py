#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from PIL import Image
import pillow_heif
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def convert_heic_to_jpg(heic_path, output_dir=None):
    """
    Convert a single HEIC file to JPG format.
    
    Args:
        heic_path (Path): Path to the HEIC file
        output_dir (Path, optional): Directory to save the JPG file. If None, uses the same directory as input
    
    Returns:
        Path: Path to the converted JPG file
    """
    try:
        # Read the HEIC file
        heif_file = pillow_heif.read_heif(str(heic_path))
        
        # Convert to PIL Image
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
        
        # Determine output path
        if output_dir is None:
            output_dir = heic_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        output_path = output_dir / f"{heic_path.stem}.jpg"
        
        # Save as JPG
        image.save(str(output_path), "JPEG", quality=95)
        return output_path
    except Exception as e:
        print(f"Error converting {heic_path}: {str(e)}")
        return None

def process_directory(input_dir, output_dir=None, max_workers=None):
    """
    Process all HEIC files in a directory and convert them to JPG.
    
    Args:
        input_dir (str): Input directory containing HEIC files
        output_dir (str, optional): Output directory for JPG files
        max_workers (int, optional): Maximum number of worker threads
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Find all HEIC files (case insensitive)
    heic_files = []
    for ext in ('*.heic', '*.HEIC'):
        heic_files.extend(input_path.glob(ext))
    
    if not heic_files:
        print("No HEIC files found in the specified directory.")
        return
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(heic_files)} HEIC files to convert...")
    
    # Convert files using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for heic_file in heic_files:
            future = executor.submit(convert_heic_to_jpg, heic_file, output_dir)
            futures.append(future)
        
        # Show progress bar
        for _ in tqdm(futures, total=len(futures), desc="Converting"):
            _.result()
    
    print("Conversion completed!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python heic_converter.py <input_directory> [output_directory]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main() 