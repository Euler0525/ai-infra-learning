import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = "What's for lunch today?"
DEVICE = "cuda"
MAX_NEW_TOKENS = 64

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
    outputs = model(**inputs, use_cache=True)

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

# Greedy Decoding
eos_token_id = tokenizer.eos_token_id
generated_token_ids = [next_token_id]
past_key_values = outputs.past_key_values
decode_attention_mask = inputs.attention_mask
stop_reason = None

print(
    f"[prefill -> token 1] "
    f"id={next_token_id}, "
    f"text={repr(tokenizer.decode([next_token_id]))}"
    "\n"
)

if next_token_id == eos_token_id:
    stop_reason = "eos"

with torch.inference_mode():
    while (
        stop_reason is None
        and len(generated_token_ids) < MAX_NEW_TOKENS
    ):
        decode_input_ids = torch.tensor(
            [[generated_token_ids[-1]]],
            dtype=inputs.input_ids.dtype,
            device=DEVICE,
        )

        decode_attention_mask = torch.cat(
            [
                decode_attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=decode_attention_mask.dtype,
                    device=DEVICE,
                ),
            ],
            dim=-1,
        )

        decode_outputs = model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Update KV Cache
        past_key_values = decode_outputs.past_key_values

        decode_last_logits = decode_outputs.logits[:, -1, :]

        # Greedy
        next_token_id = (
            decode_last_logits.argmax(dim=-1).item()
        )

        generated_token_ids.append(next_token_id)

        print(
            f"[decode -> token {len(generated_token_ids)}]\t"
            f"input_shape={list(decode_input_ids.shape)}, "
            f"logits_shape={list(decode_outputs.logits.shape)}, "
            f"id={next_token_id}, "
            f"text={repr(tokenizer.decode([next_token_id]))}"
        )

        if next_token_id == eos_token_id:
            stop_reason = "eos"

if stop_reason is None:
    stop_reason = "max_new_tokens"

generated_text = tokenizer.decode(
    generated_token_ids,
    skip_special_tokens=True,
)

print()
print("generated token ids:\n", generated_token_ids)
print()
print("generated text:\n", generated_text)
print()
print("stop reason:", stop_reason)
print()


reference = {
    "gpu_name": torch.cuda.get_device_name(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "transformers_version": transformers.__version__,
    "model_id": MODEL_ID,
    "revision": REVISION,
    "prompt": PROMPT,
    "rendered_prompt": rendered_prompt,
    "input_ids": inputs.input_ids.cpu().tolist(),
    "input_shape": list(inputs.input_ids.shape),
    "full_logits_shape": list(outputs.logits.shape),
    "last_logits_shape": list(last_logits.shape),
    "top_k": top_k,
    "generated_token_ids": generated_token_ids,
    "generated_text": generated_text,
    "stop_reason": stop_reason,
    "max_new_tokens": MAX_NEW_TOKENS,
    "peak_allocated_mib": round(
        torch.cuda.max_memory_allocated() / 1024**2,
        2,
    ),
    "eos_token_id": tokenizer.eos_token_id,
    "prompt_tokens": inputs.input_ids.shape[-1],
    "generated_tokens": len(generated_token_ids),
}

output_path = (
    Path(__file__).parent
    / "fixtures"
    / "01_forward.json"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    json.dumps(
        reference,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print("reference saved to:", output_path)


"""
Tokenizer
    ↓
Chat Template
    ↓
Token IDs
    ↓
Prefill
    ├── Full logits
    ├── Next-token prediction
    └── KV Cache
           ↓
        Decode
           ↓
        Greedy argmax
           ↓
        Next token
           ↓
       Update KV Cache
           ↓
        Decode...
           ↓
    EOS / max_new_tokens
           ↓
          Stop
"""
