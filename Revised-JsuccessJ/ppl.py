import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from tqdm import tqdm
import json

sys.path.append('/data/jaesunghwang/lm-watermarking')

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 모델 및 토크나이저 설정
model_name = "facebook/opt-1.3b"
cache_dir = "/data/huggingface_models"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

output_file = 'output_practice_eng500_beam.json'

# 퍼플렉서티 계산 함수 정의
def calculate_perplexity(text, model, tokenizer):
    if not text:
        return None
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

# 퍼플렉서티 계산 및 결과 저장
with open(output_file, 'r') as file:
    generated_texts = [json.loads(line) for line in file]

perplexity_output_file = 'perplexity_results_eng_500.json'
perplexity_values = []
with open(perplexity_output_file, 'w') as out_file:
    for idx, entry in tqdm(enumerate(generated_texts), total=len(generated_texts), desc="Calculating perplexities"):
        generated_text = entry["Generated"]
        perplexity = calculate_perplexity(generated_text, model, tokenizer)
        if perplexity is not None:
            perplexity_values.append(perplexity)  # 퍼플렉서티 값을 리스트에 추가합니다.
            json.dump({"line": idx + 1, "perplexity": perplexity, "Generated": generated_text}, out_file)
            out_file.write('\n')

print(f"Perplexity results saved to '{perplexity_output_file}'")

# 최종 평균 퍼플렉서티 계산 및 저장
if perplexity_values:
    average_perplexity = sum(perplexity_values) / len(perplexity_values)
else:
    average_perplexity = float('nan')  # 퍼플렉서티 값이 없을 경우 NaN을 반환

average_perplexity_output_file = 'average_perplexity_eng_500.json'

with open(average_perplexity_output_file, 'w') as avg_file:
    json.dump({"average_perplexity": average_perplexity}, avg_file)

print(f"Average perplexity saved to '{average_perplexity_output_file}'")
