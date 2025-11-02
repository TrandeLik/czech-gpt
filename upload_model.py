import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from src.model import CzechGPTConfig, CzechGPTModel
from tokenizers import Tokenizer
from huggingface_hub import login

AutoConfig.register("czech-gpt", CzechGPTConfig)
AutoModel.register(CzechGPTConfig, CzechGPTModel)

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

MODEL_NAME = 'vocab_64_embed384_muon120k_no_tying'
LOCAL_MODEL_PATH = f"checkpoint_{MODEL_NAME}"

NEW_REPO_ID = f"TrandeLik/czech_gpt_{MODEL_NAME}"

model = AutoModel.from_pretrained(
    LOCAL_MODEL_PATH, 
    trust_remote_code=True
)

tokenizer = AutoModel.from_pretrained(LOCAL_MODEL_PATH)

model.push_to_hub(
    repo_id=NEW_REPO_ID,
    commit_message="Upload trained model weights",
    safe_serialization=False,
)

tokenizer.push_to_hub(
    repo_id=NEW_REPO_ID,
    commit_message="Upload tokenizer",
    safe_serialization=False,
)

print("\n✅ Upload complete!")
print(f"Your model is now available at: https://huggingface.co/{NEW_REPO_ID}")

