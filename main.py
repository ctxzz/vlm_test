import json
import os
import mlx.core as mx
from mlx_vlm import load, generate

model_path = "mlx-community/llava-phi-3-mini-8bit"

model, processor = load(model_path)

user_prompt =  '''
Describe the objects in the input image. The output should follow the JSON format below, filling in the "name" and "description" values, respectively. If there are multiple objects in the image, add them to the array.

Caution:
- Output only be JSON data starting with "{" and ending with "}".
- "name" must be in lowercase.
- "description" must be no more than 100 words.

Example Output:
{
  "objects": [
    {
      "name": "string",
      "description": "string"
    }
  ]
}

Input Image: <image>

Output:
'''

prompt = processor.tokenizer.apply_chat_template(
    [{"role": "user", "content": user_prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

image_dir = "figs/"

results = []

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")])

total_images = len(image_files)

# フォルダ内のすべての画像に対して処理を実行
for idx, filename in enumerate(image_files):
    image_path = os.path.join(image_dir, filename)
    
    print(f"Processing image {idx + 1} of {total_images}: {filename}")
    
    output = generate(
        model, processor, image_path, prompt, temp=0.8, verbose=False, max_tokens=1024
    )

    output_text = output.strip()

    try:
        output_json = json.loads(output_text)
    except json.JSONDecodeError:
        formatted_output_text = output_text.replace('\\n', '\n').replace('\\t', '\t')
        try:
            output_json = json.loads(formatted_output_text)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON for image {idx + 1}: {filename}")
            print(f"Output: {output_text}")
            output_json = {"error": "Failed to decode JSON"}

    results.append({"index": idx, "filename": filename, "output": output_json})

with open("output.json", "w") as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

print("Output saved to output.json")
