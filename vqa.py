from llm_prompting import LLM
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from PIL import Image

# load processor and model
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
QW = LLM(model_name)

# Load the image locally
image_path = "file:///Users/ricardobaptista/Documents/Research/LLMPrompting/LLMExamples/NS.png"
#image_path = "https://www.fdy.tu-darmstadt.de/media/fachgebiet_fdy/fdy_forschung_bilder/area_compressible_1300x0.png"
image_size = 128**2
#prompt = "Describe what you see in this image </img>."
prompt =  "Describe this image representing the solution to Navier Stokes equation. Does the field appear to be turbulent? </img>."

# Add user input to conversation
messages = [
   {
       "role": "user",
       "content": [
           {"type": "image",
            "image": image_path,
            "min_pixels": image_size,
            "max_pixels": image_size,
           },
           {"type": "text", "text": prompt}, 
       ],
   }
]

# process inputs
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# # Inference: Generation of the output
generated_ids = QW.model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

