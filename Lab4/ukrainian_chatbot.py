from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from huggingface_hub import login
import os
import requests
import warnings
import time
from huggingface_hub import hf_hub_download

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

def download_with_retry(model_id, filename, token, max_retries=3):
    """
    Download model files with retry logic.
    """
    for attempt in range(max_retries):
        try:
            print(f"\nAttempting download ({attempt + 1}/{max_retries})...")
            return hf_hub_download(
                repo_id=model_id,
                filename=filename,
                token=token,
                local_files_only=False,
                resume_download=True,
                max_retries=2,
                legacy_cache_layout=False
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # backoff
                print(f"Download failed. Waiting {wait_time} seconds before retrying...")
                print(f"Error: {str(e)}")
                time.sleep(wait_time)
            else:
                raise e

def check_system_requirements():
    """
    Check if system meets basic requirements.
    """
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        print(f"\nSystem information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {total_memory:.2f} GB")
        print(f"PyTorch version: {torch.__version__}")
    except Exception as e:
        print(f"Could not get complete system information: {e}")

def initialize_model():
    """
    Initialize the smaller 1.7B model and tokenizer with authentication.
    """
    print("Initializing smaller 1.7B model... This may take a moment.")
    
    auth_token = os.getenv("HUGGING_FACE_TOKEN")
    if not auth_token:
        print("Error: HUGGING_FACE_TOKEN not found in .env file")
        print("Please ensure you have a .env file with your token")
        sys.exit(1)
    
    # Create cache directory if it doesn't exist
    os.makedirs("model_cache", exist_ok=True)
    
    try:
        print("\nChecking internet connection...")
        requests.get("https://huggingface.co", timeout=5)
        
        # Login with the token from .env
        login(auth_token)
        
        # Use the smaller 1.7B instruct model
        model_id = "utter-project/EuroLLM-1.7B-Instruct"
        
        print("Downloading and initializing tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=auth_token,
                use_fast=True
            )
        except Exception:
            print(f"Error downloading tokenizer. Retrying with download_with_retry...")
            download_with_retry(model_id, "tokenizer.json", auth_token)
            download_with_retry(model_id, "tokenizer_config.json", auth_token)
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=auth_token,
                use_fast=True,
                local_files_only=True
            )
        
        # If no dedicated pad token exists, create one to avoid warnings
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        print("\nDownloading and initializing the 1.7B model (this may take some time)...")
        model_kwargs = {
            "token": auth_token,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU...")
            # Adjust memory if needed
            model_kwargs.update({
                "device_map": "auto",
                "max_memory": {0: "10GB"}  # reduce for a smaller model
            })
        else:
            print("CUDA not available. Using CPU...")
            model_kwargs.update({
                "device_map": "cpu",
                "offload_folder": "model_cache"
            })
            
        # Attempt to load the model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
        except Exception as e:
            print(f"Error downloading model. Retrying with download_with_retry...")
            # If the model is sharded, adapt filenames as needed
            # (Here we assume 1 or more shards)
            for i in range(1, 3):  # Example: if it has 2 shards, or adapt to your case
                filename = f"model-0000{i}-of-00002.safetensors"
                download_with_retry(model_id, filename, auth_token)
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=True,
                **model_kwargs
            )
        
        # Resize token embeddings if a new pad token was added
        if tokenizer.pad_token_id is not None:
            model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
        
    except requests.exceptions.ConnectionError:
        print("Error: No internet connection. Please check your connection and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during model initialization: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your available RAM (at least 8-10GB recommended)")
        print("2. Ensure you have enough disk space (model is smaller but still large)")
        print("3. Verify your Hugging Face token is correct")
        print("4. Try running the script with administrator privileges")
        sys.exit(1)

def generate_response(model, tokenizer, user_input, conversation_history):
    """
    Generate a response using the smaller 1.7B model.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are EuroLLM --- an AI assistant that speaks Ukrainian only. "
                "If user prompts you in any other language than Ukrainian you must reply 'Sorry, this chat bot speaks only Ukrainian' in the prompted language "
                "If user prompts you in Russian you must reply 'Слава Україні!'"
            )
        }
    ] + conversation_history + [
        {"role": "user", "content": user_input}
    ]

    try:
        # If apply_chat_template() returns a single Tensor of input_ids:
        # we create an attention_mask manually.
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        # Create attention mask (1 = actual token, 0 = padding)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Optional: if your conversation style includes "Assistant:" in the text
        if "assistant:" in response:
            response = response.split("assistant:")[-1].strip()
        
        return response
    except torch.cuda.OutOfMemoryError:
        return (
            "Вибачте, виникла помилка через нестачу пам'яті. Спробуйте коротше повідомлення."
        )
    except Exception as e:
        return f"Помилка генерації відповіді: {str(e)}"

def cleanup():
    """
    Cleanup function to free memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """
    Main function to run the chatbot with the smaller 1.7B model.
    """
    try:
        print("Ласкаво просимо до українського чатбота!")
        
        check_system_requirements()
        model, tokenizer = initialize_model()
        conversation_history = []
        
        print("\nМодель готова! Починайте спілкування.")
        print("(Введіть 'вийти' для завершення розмови)\n")
        
        while True:
            user_input = input("\nВи: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['вийти', 'exit', 'quit']:
                print("\nДо побачення!")
                break
            
            print("\nОбробка відповіді...")
            response = generate_response(model, tokenizer, user_input, conversation_history)
            print("\nБот:", response)
            
            conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])
            
            # Keep conversation history manageable
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
    except KeyboardInterrupt:
        print("\n\nРоботу програми перервано. До побачення!")
    except Exception as e:
        print(f"\nКритична помилка: {str(e)}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
