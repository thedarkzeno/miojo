from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

config = LlamaConfig(hidden_size=768, num_hidden_layers=12)
model = LlamaForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

model.save_pretrained("checkpoints/dummy_llama")
tokenizer.save_pretrained("checkpoints/dummy_llama")