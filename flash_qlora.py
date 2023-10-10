from llama_attn_hijack_flash import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

if __name__ == '__main__':
    from train import train
    train()
