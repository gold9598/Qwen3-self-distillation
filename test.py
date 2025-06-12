from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

teacher_name = "Qwen/Qwen3-1.7B"

teacher = AutoModelForCausalLM.from_pretrained(
    teacher_name,
    trust_remote_code=True,
    device_map="auto",
    max_memory={0: "0GB", 1: "80GB", 2: "80GB"},
    use_cache=False,
)
