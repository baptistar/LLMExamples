from transformers import AutoTokenizer, AutoModelForCausalLM

# load model
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name) 

# run without explanation
messages = [
    {"role": "user", "content": "A man named Frank travels from NYC to Philadelphia by road, on a Monday, leaving at 9:00am and arriving at 12noon. He then travels from Philadelphia to NYC the next day, Tuesday, also leaving at 9:00am and arriving at 12 noon. He takes the same route, but on Tuesday reverses the route taken on Monday. Is it necessarily the case that there is at least one point that he will be located at the same time, albeit on the two different days? Just give a Yes or No answer."}]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
print('===== Yes/No Answer ======')
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))  

# add explanation
messages = [
    {"role": "user", "content": "A man named Frank travels from NYC to Philadelphia by road, on a Monday, leaving at 9:00am and arriving at 12noon. He then travels from Philadelphia to NYC the next day, Tuesday, also leaving at 9:00am and arriving at 12 noon. He takes the same route, but on Tuesday reverses the route taken on Monday. Is it necessarily the case that there is at least one point that he will be located at the same time, albeit on the two different days? Explain your answer."}]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
print('===== With Reasoning ======')
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))  
