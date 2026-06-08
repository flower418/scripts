import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with the base model or a LoRA checkpoint.")
    parser.add_argument("--model", default=".", help="Base model path.")
    parser.add_argument("--adapter", default=None, help="LoRA adapter/checkpoint path, e.g. ./sft_22k/checkpoint-200.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def build_inputs(tokenizer, messages, device):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return tokenizer(prompt, return_tensors="pt").to(device)

    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        prompt += f"{role}: {content}\n"
    prompt += "assistant: "
    return tokenizer(prompt, return_tensors="pt").to(device)


def main():
    args = parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded LoRA adapter: {adapter_path}")

    model.eval()
    device = next(model.parameters()).device
    messages = []

    print("Chat ready. Type /exit to quit, /clear to reset history.")
    while True:
        try:
            user_text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "exit", "quit", "q"}:
            break
        if user_text == "/clear":
            messages.clear()
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": user_text})
        inputs = build_inputs(tokenizer, messages, device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"\n助手: {answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
