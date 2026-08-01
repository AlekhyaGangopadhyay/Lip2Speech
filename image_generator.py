import os
import sys

# Crucial: Configure Hugging Face to use D drive for downloads before importing diffusers/transformers
HF_CACHE_DIR = "D:/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except ImportError:
    pass


import torch

# Try loading diffusers dynamically
AutoPipelineForText2Image = None

def expand_prompt(text):
    """
    Expands structured GRID corpus commands into rich visual prompts for SD-Turbo.
    Example: "place blue in at g two now" -> 
             "A clean studio photo of a hand placing a solid blue block at grid coordinate G2, minimalist background, 8k resolution, sharp focus"
    """
    if not text:
        return "A simple abstract geometric pattern"
    
    words = text.lower().split()
    
    # Detect command/action
    action = "place"
    for w in ["bin", "lay", "place", "set"]:
        if w in words:
            action = w
            break
            
    # Detect color
    color = "blue"
    for w in ["blue", "green", "red", "white"]:
        if w in words:
            color = w
            break
            
    # Detect coordinate letter and number word
    letters = [chr(i) for i in range(ord('a'), ord('z')+1)]
    number_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
    
    coord_letter = None
    coord_number = None
    
    for i, w in enumerate(words):
        if w in number_words:
            coord_number = w
            # Check if the previous word was a letter
            if i > 0 and words[i-1] in letters:
                coord_letter = words[i-1]
                
    # Fallback search if not adjacent
    if not coord_letter:
        for w in words:
            if w in letters and w != "a":  # prefer letters other than 'a'
                coord_letter = w
                break
        else:
            if "a" in words:
                coord_letter = "a"
                
    if not coord_number:
        for w in words:
            if w in number_words:
                coord_number = w
                break
                
    # Convert mappings
    action_desc = {
        "place": "a hand placing",
        "set": "a hand setting down",
        "lay": "a hand laying down",
        "bin": "a hand putting in a bin"
    }.get(action, "a hand placing")
    
    color_desc = color
    
    num_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }
    num_digit = num_map.get(coord_number, "")
    
    if coord_letter and coord_number:
        coord_str = f"at grid coordinate {coord_letter.upper()}{num_digit}"
    elif coord_letter:
        coord_str = f"at grid coordinate {coord_letter.upper()}"
    elif coord_number:
        coord_str = f"at position {num_digit}"
    else:
        coord_str = "on a grid table"
        
    prompt = f"A clean studio photo of {action_desc} a solid {color_desc} block {coord_str}, minimalist background, 8k resolution, sharp focus"
    return prompt

def generate_image_from_text(text, output_path):
    """
    Expands the text prompt and generates an image using local stabilityai/sd-turbo.
    Downloads the model to D:/hf_cache if not already cached.
    """
    global AutoPipelineForText2Image
    if AutoPipelineForText2Image is None:
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as e:
            print("[ERROR] diffusers library is not installed or could not be loaded.")
            raise e
        
    prompt = expand_prompt(text)
    print(f"  [Image Generator] Original: '{text}'")
    print(f"  [Image Generator] Expanded Prompt: '{prompt}'")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [Image Generator] Using device: {device}")
    
    # Configure datatype & variant
    dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if device == "cuda" else None
    
    print(f"  [Image Generator] Loading 'stabilityai/sd-turbo' (Cache dir: {HF_CACHE_DIR}) ...")
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=dtype,
        variant=variant,
        cache_dir=HF_CACHE_DIR
    )
    
    pipe.to(device)
    
    print(f"  [Image Generator] Generating image (1 step)...")
    # SD-Turbo recommends 1 step and guidance_scale=0.0
    result = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0)
    image = result.images[0]
    
    # Save the output image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    image.save(output_path)
    print(f"  [Image Generator] Image successfully saved to: {output_path}")
    return output_path
