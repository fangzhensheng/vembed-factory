import os
import sys
from huggingface_hub import snapshot_download

# Configuration
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
OUTPUT_DIR = "models/dinov3-vitb16-pretrain-lvd1689m"

def main():
    print(f"Starting download of {MODEL_ID}...")
    print(f"Target directory: {os.path.abspath(OUTPUT_DIR)}")
    
    try:
        # Ensure the directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Download the model
        # Using token=True will use the locally saved token (from `huggingface-cli login`)
        # If HF_TOKEN env var is set, it will be used instead.
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=True 
        )
        print("\nDownload completed successfully!")
        
        # Verify critical files
        expected_files = ["config.json", "model.safetensors"]
        files = os.listdir(OUTPUT_DIR)
        for f in expected_files:
            if f in files:
                print(f"Verified: {f} exists.")
            elif f == "model.safetensors" and "pytorch_model.bin" in files:
                print("Verified: pytorch_model.bin exists.")
            else:
                print(f"Warning: {f} not found in output directory.")
                    
    except Exception as e:
        error_msg = str(e)
        print(f"\nError occurred: {error_msg}")
        
        if "403" in error_msg or "gated" in error_msg.lower() or "authorized" in error_msg.lower():
            print("\n" + "="*60)
            print("ACCESS DENIED: This model is gated.")
            print("="*60)
            print(f"You need to accept the license agreement to access '{MODEL_ID}'.")
            print(f"Please visit: https://huggingface.co/{MODEL_ID}")
            print("1. Log in to Hugging Face.")
            print("2. Click 'Agree and access repository'.")
            print("3. Once approved, run this script again.")
            print("="*60)
        elif "401" in error_msg:
             print("\n" + "="*60)
             print("AUTHENTICATION FAILED.")
             print("="*60)
             print("Please log in using: huggingface-cli login")
             print("Or set HF_TOKEN environment variable.")
             print("="*60)
             
        sys.exit(1)

if __name__ == "__main__":
    main()
