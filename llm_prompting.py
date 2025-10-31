from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration

class LLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name)
        if 'VL' in model_name:
            self.model      = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name)
        else:
            self.model      = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.max_tokens = 500

    def generate(self, messages, verbose=True):
        inputs = self.tokenizer.apply_chat_template(
        	messages,
        	add_generation_prompt=True,
        	tokenize=True,
        	return_dict=True,
        	return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=500)
        reply = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        if verbose == True:
            print(self.model_name+': ', reply)
        return reply

    def multi_turn(self):
        messages = []
        while True:
            # get input
            user_input = input("You: ")
            if user_input.lower() in {"quit", "exit"}:
                print("Chat ended.")
                break 
            # Add user input to conversation
            messages.append({"role": "user", "content": user_input})
            # call model 
            reply = self.generate(messages)
            #  Add assistant reply back to history
            messages.append({"role": "assistant", "content": reply})


