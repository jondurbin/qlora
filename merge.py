from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--peft", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--push", action="store_true")

    return parser.parse_args()

def main():
    args = get_args()
    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Loading PEFT: {args.peft}")
    model = PeftModel.from_pretrained(base_model, args.peft)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    if args.push:
        print(f"Saving to hub ...")
        model.push(args.out, use_temp_dir=False)
        tokenizer.push(args.out, use_temp_dir=False)

if __name__ == "__main__" :
    main()
