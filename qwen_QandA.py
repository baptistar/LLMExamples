from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name) 

messages = [
    {"role": "user", "content": "Explain Bayes’ theorem simply without formulas."},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))  


messages = [
    {"role": "user", "content": "Explain Bayes’ theorem simply without formulas."},
    {"role": "assistant", "content": "Bayes’ theorem is a way to update our beliefs about something based on new information. It helps us figure out how likely an event is, given some evidence"},
    {"role": "user", "content": "Now explain it more formally to a mathematician."}
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))  
