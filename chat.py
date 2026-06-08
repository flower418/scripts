import argparse
from pathlib import Path
from threading import Thread

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with the base model or a LoRA checkpoint.")
    parser.add_argument("--model", default=".", help="Base model path.")
    parser.add_argument("--adapter", default=None, help="LoRA adapter/checkpoint path, e.g. ./sft_22k/checkpoint-200.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Device to run inference on.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)

    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.to(device)
        print(f"Loaded LoRA adapter: {adapter_path}")

    model.eval()
    messages = []

    print(f"Chat ready on {device}. Type /exit to quit, /clear to reset history.")
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

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        print("\n助手: ", end="", flush=True)
        with torch.no_grad():
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            chunks = []
            for chunk in streamer:
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            thread.join()
        print()

        answer = "".join(chunks).strip()
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
