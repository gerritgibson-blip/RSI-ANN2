"""Rebuild full weights: python merge.py <out_dir>"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "Qwen/Qwen2.5-Coder-7B-Instruct"
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, "model/adapter").merge_and_unload()
m.save_pretrained(sys.argv[1]); AutoTokenizer.from_pretrained(base).save_pretrained(sys.argv[1])
