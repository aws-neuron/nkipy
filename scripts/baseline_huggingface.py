import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai/gpt-oss-120b"

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# all Qwen3 models use the same tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
with open("nki_prompt_10k.txt", "r") as f:
    prompt = f.read()
model_input = tokenizer(prompt, return_tensors="pt")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # device_map="cpu",
    device_map="auto",
)
# model.save_pretrained("./Qwen3-30B-A3B")
#model.parameters()

# prompt = "The capital of France is"
# print(f"{model_input['input_ids'].size()=}")
generated_id = model.generate(
    **model_input,
    max_new_tokens=2000,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
)
output_ids = generated_id[0][len(model_input.input_ids[0]):].tolist() 
output = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
print(f"{output=}")
# output = " \n\nThe example problem is 102_Standard_matrix_add. Numpy"
