import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = "What's for lunch today?"
DEVICE = "cuda"

if not torch.cuda.is_available():
    raise RuntimeError("NO available NVIDIA GPU with CUDA.")

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    dtype=torch.bfloat16,
    local_files_only=True,
).to(device=DEVICE)
model.eval()

# Eval
messages = [
    {"role": "user", "content": PROMPT},
]

print("message:\n", messages)
print()

rendered_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("rendered_prompt:\n", rendered_prompt)

inputs = tokenizer(
    rendered_prompt,
    return_tensors="pt",
).to(device=DEVICE)

print("inputs:\n", inputs, "\t", inputs.input_ids.shape)
print()

token_ids = inputs.input_ids[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(token_ids)

print("token                    -> token id:")
i = 1
for token, token_id in zip(tokens, token_ids):
    print(f"[{i:2d}]{repr(token):20s} -> {token_id}")
    i += 1
print()

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    outputs = model(**inputs, use_cache=False)

# [batch, sequence, vocabulary]  Qwen2.5-0.5B 的词表大小是 151936，所以每个序列位置都会得到 151936 个候选分数
print("outputs:\n", outputs, "\t", outputs.logits.shape)
print()

last_logits = outputs.logits[:, -1, :]
print("last_logits:\n", last_logits, "\t", last_logits.shape)
print()

next_token_id = last_logits.argmax(dim=-1).item()
top_values, top_indices = torch.topk(last_logits[0], k=5)
top_k = [
    {
        "token_id": token_id,
        "logit": logit,
        "text": tokenizer.decode([token_id]),
    }
    for token_id, logit in zip(
        top_indices.tolist(),
        top_values.tolist(),
    )
]
print("top 5:\n", top_k)
print()

print("next token:\n", tokenizer.decode([next_token_id]))
print()

reference = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "prompt": PROMPT,
    "rendered_prompt": rendered_prompt,
    "input_ids": inputs.input_ids.cpu().tolist(),
    "input_shape": list(inputs.input_ids.shape),
    "full_logits_shape": list(outputs.logits.shape),
    "last_logits_shape": list(last_logits.shape),
    "top_k": top_k,
    "next_token_id": next_token_id,
    "next_token_text": tokenizer.decode([next_token_id]),
    "peak_allocated_mib": round(
        torch.cuda.max_memory_allocated() / 1024**2,
        2,
    ),
}

print("peak allocated MiB:", reference["peak_allocated_mib"])

"""
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 12927.36it/s]
message:
 [{'role': 'user', 'content': "What's for lunch today?"}]

rendered_prompt:
 <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
What's for lunch today?<|im_end|>
<|im_start|>assistant

inputs:
 {'input_ids': tensor([[151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
            553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
             13, 151645,    198, 151644,    872,    198,   3838,    594,    369,
          15786,   3351,     30, 151645,    198, 151644,  77091,    198]],
       device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}    torch.Size([1, 35])

token                    -> token id:
[ 1]'<|im_start|>'       -> 151644
[ 2]'system'             -> 8948
[ 3]'Ċ'                  -> 198
[ 4]'You'                -> 2610
[ 5]'Ġare'               -> 525
[ 6]'ĠQ'                 -> 1207
[ 7]'wen'                -> 16948
[ 8]','                  -> 11
[ 9]'Ġcreated'           -> 3465
[10]'Ġby'                -> 553
[11]'ĠAlibaba'           -> 54364
[12]'ĠCloud'             -> 14817
[13]'.'                  -> 13
[14]'ĠYou'               -> 1446
[15]'Ġare'               -> 525
[16]'Ġa'                 -> 264
[17]'Ġhelpful'           -> 10950
[18]'Ġassistant'         -> 17847
[19]'.'                  -> 13
[20]'<|im_end|>'         -> 151645
[21]'Ċ'                  -> 198
[22]'<|im_start|>'       -> 151644
[23]'user'               -> 872
[24]'Ċ'                  -> 198
[25]'What'               -> 3838
[26]"'s"                 -> 594
[27]'Ġfor'               -> 369
[28]'Ġlunch'             -> 15786
[29]'Ġtoday'             -> 3351
[30]'?'                  -> 30
[31]'<|im_end|>'         -> 151645
[32]'Ċ'                  -> 198
[33]'<|im_start|>'       -> 151644
[34]'assistant'          -> 77091
[35]'Ċ'                  -> 198

outputs:
 CausalLMOutputWithPast(loss=None, logits=tensor([[[ 3.8750,  7.7812,  3.2969,  ..., -1.0781, -1.0781, -1.0781],
         [ 5.4062, 11.2500,  7.2500,  ...,  0.1768,  0.1768,  0.1768],
         [ 7.4062, 10.6250, 13.1875,  ...,  0.3008,  0.3008,  0.3008],
         ...,
         [ 1.9766,  5.2500,  7.6250,  ..., -1.1641, -1.1641, -1.1641],
         [ 3.0000,  9.1250,  1.7344,  ..., -3.1875, -3.1875, -3.1875],
         [ 7.3438, 14.0625,  7.1250,  ..., -3.1094, -3.1094, -3.1094]]],
       device='cuda:0', dtype=torch.bfloat16), past_key_values=None, hidden_states=None, attentions=None)        torch.Size([1, 35, 151936])

last_logits:
 tensor([[ 7.3438, 14.0625,  7.1250,  ..., -3.1094, -3.1094, -3.1094]],
       device='cuda:0', dtype=torch.bfloat16)    torch.Size([1, 151936])

top 5:
 [{'token_id': 2121, 'logit': 19.625, 'text': 'As'}, {'token_id': 40, 'logit': 19.0, 'text': 'I'}, {'token_id': 15364, 'logit': 18.375, 'text': 'Today'}, {'token_id': 43, 'logit': 17.875, 'text': 'L'}, {'token_id': 9707, 'logit': 17.75, 'text': 'Hello'}]

next token:
 As

peak allocated MiB: 988.73
"""
