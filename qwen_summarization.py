from llm_prompting import LLM

# define model
model_name = "Qwen/Qwen2.5-3B-Instruct"
QW = LLM(model_name)

# Add user input to conversation
user_input = "Summarize the following introduction section of a paper into a single sentence: "
intro_file = "./introduction.txt"
with open(intro_file, "r", encoding="utf-8") as file:
    for line in file:
        user_input += line  # append each line to the string
print(user_input)

messages = [{"role": "user", "content": user_input}]
reply = QW.generate(messages)
