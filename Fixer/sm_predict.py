import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
from tqdm import tqdm


def get_lora_model(base_model_path,
                   lora_model_input_path,
                   lora_model_output_path):
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto",
                                                 trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_model_input_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(lora_model_output_path, max_shard_size="2048MB", safe_serialization=True)


    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    tokenizer.save_pretrained(lora_model_output_path)



def get_model_result(jsonl_path, output_path, base_model_path, fintune_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    device = "cuda"

    fintune_model = AutoModelForCausalLM.from_pretrained(
        fintune_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    ).eval()

    def get_result(model_inputs, model):
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=128
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            conversation = data["conversations"][0]
            prompt = conversation["value"]
            groundtruth = data["conversations"][1]["value"]
            id_ = data["id"]

            messages = [
                {"role": "system", "content": "You are a software engineer."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text],truncation=True,max_length=1024,return_tensors="pt").to(device)

            fintune_model_response = get_result(model_inputs, fintune_model)

            print("after tuning:", fintune_model_response)

            results.append({
                "id": id_,
                "prompt": prompt,
                "groundtruth": groundtruth,
                "finetune_result": fintune_model_response
            })

        with open(output_path, "w", encoding="utf-8") as f_out:
            for result in results:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Results saved to {output_path}")


if __name__ == '__main__':
    base_path = "/root/autodl-tmp/Qwen/Qwen2___5-Coder-14B-Instruct"
    lora_in_path = "../output/qwen2cins_diff_lora_5epoch_enhance"
    lora_out_path = "/root/autodl-tmp/qwenins5epochenhance/"
    jsonl_input_path = "../data/fixer_test_diff_conv.jsonl" 
    jsonl_output_path = "../data/fixer_lora_epoch5_diff_enhance_result.jsonl"  
    if not os.path.exists(lora_out_path):
        get_lora_model(base_path, lora_in_path, lora_out_path)
    get_model_result(jsonl_input_path, jsonl_output_path, base_path, lora_out_path)
