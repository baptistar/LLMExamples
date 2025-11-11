from llm_prompting import LLM

# define model
model_name = "Qwen/Qwen2.5-3B-Instruct"
QW = LLM(model_name)

# Add user input to conversation
user_input = "Translate ``Thank you for your help with answering my math question.'' into French" 
messages = [{"role": "user", "content": user_input}]
reply = QW.generate(messages)

#  Add assistant reply back to history
messages.append({"role": "assistant", "content": reply})
user_input = "Translate ``Merci pour votre aide à répondre à ma question de mathématiques'' into English"
messages = [{"role": "user", "content": user_input}]
reply = QW.generate(messages)

