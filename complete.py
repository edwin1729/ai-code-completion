# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
model = AutoModelForCausalLM.from_pretrained("bigcode/tiny_starcoder_py")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'def sort(',
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=100,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
