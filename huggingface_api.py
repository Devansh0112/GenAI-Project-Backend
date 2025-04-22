import os
import requests
import json
from PIL import Image
import io
import base64
from utils import image_to_base64, base64_to_image

class HuggingFaceAPI:
    """Service to interact with Hugging Face Inference API"""
    
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4", api_token=None):
        # Try getting token from parameter first
        self.api_token = api_token
        
        # If not provided, try environment variable
        if not self.api_token:
            self.api_token = os.environ.get("HF_API_TOKEN")
        
        # If still not found, try reading from config file
        if not self.api_token:
            try:
                if os.path.exists('config.json'):
                    with open('config.json', 'r') as f:
                        config = json.load(f)
                        self.api_token = config.get('hf_api_token')
            except:
                pass
                
        # If still not found, raise error
        if not self.api_token:
            raise ValueError(
                "Hugging Face API token not found. Please provide it through one of these methods:\n"
                "1. Pass directly to the HuggingFaceAPI constructor\n"
                "2. Set the HF_API_TOKEN environment variable\n"
                "3. Create a config.json file with an 'hf_api_token' field"
            )
        
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def generate_image(
        self,
        prompt,
        negative_prompt="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None,
    ):
        """Generate an image from a text prompt using HF API"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
        }
        
        # Add seed if provided
        if seed is not None:
            payload["parameters"]["seed"] = seed
        
        # Make API request
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        # The response is the binary image data
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to base64 for API response
        base64_image = image_to_base64(image)
        
        # Get the seed used (if provided in response, otherwise use input seed or None)
        result_seed = seed
        if "seed" in response.headers:
            result_seed = int(response.headers.get("seed"))
        
        return {
            "image": base64_image,
            "prompt": prompt,
            "seed": result_seed,
        }
    
    def generate_variations(
        self,
        image,
        prompt="",
        negative_prompt="",
        strength=0.75,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_variations=4,
    ):
        """Generate variations of an input image using img2img"""
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # For img2img we need to use a different endpoint
        img2img_url = f"https://api-inference.huggingface.co/models/{self.model_id}/img2img"
        
        variations = []
        
        # Generate multiple variations
        for _ in range(num_variations):
            files = {
                'image': img_byte_arr,
            }
            
            data = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'strength': strength,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_inference_steps,
            }
            
            # Make API request
            response = requests.post(img2img_url, headers=self.headers, files=files, data=data)
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
            # The response is the binary image data
            variation_image = Image.open(io.BytesIO(response.content))
            
            # Convert to base64 for API response
            base64_image = image_to_base64(variation_image)
            
            # Get seed if available in response headers, otherwise generate a random one
            seed = None
            if "seed" in response.headers:
                seed = int(response.headers.get("seed"))
            else:
                import random
                seed = random.randint(0, 2**32 - 1)
            
            variations.append({
                "image": base64_image,
                "prompt": prompt,
                "seed": seed,
            })
        
        return variations