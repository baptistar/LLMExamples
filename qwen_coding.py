from llm_prompting import LLM

user_input = "Write Matlab code to numerically solve a nonlinear ODE with vector field f. Only include the code in the response."
messages = [{"role": "user", "content": user_input}]

model_name = "Qwen/Qwen2.5-3B"
QW = LLM(model_name)
QW.generate(messages)

model_name = "Qwen/Qwen2.5-Coder-3B"
QW = LLM(model_name)
QW.generate(messages)

