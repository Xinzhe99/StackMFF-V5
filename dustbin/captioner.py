import os
import dashscope
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

print("Using DashScope API for image captioning with concurrent requests.")

# Directory containing images
image_dir = "/home/ot/Downloads/1756654037_X1nzhe99/DIODE-5000/TR/AiF"

# Supported image extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

# Get list of image files that do not have corresponding .txt files
files = [p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in image_extensions and not p.with_suffix('.txt').exists()]

def process_image(img_path):
    start_time = time.time()
    
    try:
        # Prepare messages for the current image
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": "file://" + str(img_path)},  # Use file:// for local images
                    {"text": "Describe this image."}
                ]
            }
        ]
        
        # Call DashScope API
        response = dashscope.MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model='qwen-vl-max-latest',
            messages=messages
        )
        
        # Check if response is valid
        if response is None:
            raise ValueError("API response is None")
        
        # Check status code if available
        if hasattr(response, 'status_code') and response.status_code != 200:
            raise ValueError(f"API call failed with status code: {response.status_code}")
        
        # Extract description from response
        if 'output' not in response or 'choices' not in response['output'] or not response['output']['choices']:
            raise ValueError("Invalid response structure")
            
        desc = response['output']['choices'][0]['message']['content'][0]['text']
        
        end_time = time.time()
        return desc, img_path, end_time - start_time
        
    except Exception as e:
        end_time = time.time()
        error_desc = f"Error processing {img_path.name}: {str(e)}"
        return error_desc, img_path, end_time - start_time

# Use ThreadPoolExecutor for concurrent requests
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_image, img_path) for img_path in files]
    
    for future in tqdm(as_completed(futures), total=len(files)):
        desc, img_path, elapsed = future.result()
        
        # Only save if description is not an error message
        if not desc.startswith("Error processing"):
            # Save the description to a .txt file with the same name
            txt_path = img_path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(desc)
            
            print(f"✅ Image: {img_path.name} | Saved to: {txt_path.name} | Time: {elapsed:.2f}s | Description: {desc}")
        else:
            print(f"❌ {desc} | Time: {elapsed:.2f}s")

print("All images processed.")