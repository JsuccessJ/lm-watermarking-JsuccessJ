import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from tqdm import tqdm
import json

sys.path.append('/data/jaesunghwang/lm-watermarking-JsuccessJ')

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model_name = "beomi/KoAlpaca-llama-1-7b"
cache_dir = "/data/huggingface_models"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# 워터마크 로짓 프로세서 초기화
watermark_processor = WatermarkLogitsProcessor(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25,
    delta=2.0,
    seeding_scheme="simple_1"
)

# 로짓 프로세서 리스트
logits_processor = LogitsProcessorList([watermark_processor])

# 입력 파일에서 데이터 읽기 (JSON 형식)
input_file = 'aggregated_titles.json'
with open(input_file, 'r', encoding='utf-8') as file:
    input_titles = [json.loads(line)["title"] for line in file]

# 결과를 저장할 파일 열기 (JSON 형식)
output_file = 'output_practice_kor500_beam.json'
with open(output_file, 'w', encoding='utf-8') as out_file:
    # 500개의 문장에 대해 각각 텍스트 생성
    for idx, title in tqdm(enumerate(input_titles[:500]), total=500, desc="Generating texts"):
        input_text = title.strip()
        
        # 입력 텍스트 토큰화
        tokenized_input = tokenizer(input_text, return_tensors='pt').to(device)
        if 'token_type_ids' in tokenized_input:
            tokenized_input.pop('token_type_ids')

        # 텍스트 생성
        output_tokens = model.generate(
            **tokenized_input,
            logits_processor=logits_processor,
            max_new_tokens=128,
            do_sample=True,
            num_beams=4,
            pad_token_id=tokenizer.eos_token_id  # pad_token_id를 eos_token_id로 설정
        )

        # 새로 생성된 토큰 분리
        output_tokens = output_tokens[:, tokenized_input["input_ids"].shape[-1]:]

        # 토큰을 텍스트로 디코딩
        output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

        # 유효한 유니코드 문자열인지 확인
        output_text = output_text.encode('utf-8', 'ignore').decode('utf-8')

        # \n\n을 띄어쓰기 한 칸으로 변경하고 앞뒤 공백 제거
        output_text = output_text.replace('\n\n', ' ').strip()
        
        # JSON 형식으로 저장
        json.dump({"Generated": output_text}, out_file, ensure_ascii=False)
        out_file.write('\n')

print("Generated texts saved to 'output_practice_kor500_beam.json'")

# 워터마크 텍스트 파일 읽기
with open(output_file, 'r', encoding='utf-8') as file:
    watermarked_texts = [json.loads(line) for line in file]

# 워터마크 탐지기 초기화
watermark_detector = WatermarkDetector(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25,
    seeding_scheme="simple_1",
    device=device,
    tokenizer=tokenizer,
    z_threshold=4.0,
    normalizers=[],
)

# 각 문장에서 워터마크 탐지
with open('detection_results_kor500.json', 'w', encoding='utf-8') as detect_file:
    for idx, entry in tqdm(enumerate(watermarked_texts), total=len(watermarked_texts), desc="Detecting watermarks"):
        watermarked_text = entry["Generated"]
        score_dict = watermark_detector.detect(text=watermarked_text)
        json.dump({"line": idx + 1, "detection_result": score_dict}, detect_file, ensure_ascii=False)
        detect_file.write('\n')

print("Watermark detection results saved to 'detection_results_kor500.json'")
