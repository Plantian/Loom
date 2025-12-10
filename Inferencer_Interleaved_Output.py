import os
import pandas as pd
import argparse
import random
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from pathlib import Path
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
# from peft import LoraConfig, get_peft_model
from inferencer import InterleaveInferencer  

def parse_args():
    parser = argparse.ArgumentParser(description='BAGEL Model Inference')
    parser.add_argument('--parquet_path', type=str, help='Path to the input Parquet file')
    parser.add_argument('--end_id', type=int, help='Starting sample ID (default: 0)')
    parser.add_argument('--model_path', type=str, help='Path to the model directory')
    parser.add_argument('--ema_model_path', type=str, help='Path to EMA model checkpoint')
    parser.add_argument('--output_dir', type=str, help='Output directory for generated images')
    parser.add_argument('--max_mem_per_gpu', type=str, default="80GiB", help='Maximum memory per GPU')
    parser.add_argument('--use_lora', type=bool, default=False, help='Whether to use LoRA')
    parser.add_argument('--visual_gen', type=bool,default=True, help='Enable visual generation')
    parser.add_argument('--visual_und', type=bool, default=True, help='Enable visual understanding')
    return parser.parse_args()

def initialize_model_and_inferencer(model_path, ema_model_path, max_mem_per_gpu, use_lora, visual_und, visual_gen):
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    
    # Bagel config preparing
    config = BagelConfig(
        visual_gen=visual_gen, visual_und=visual_und, llm_config=llm_config, vit_config=vit_config,
        vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh', latent_patch_size=2, max_latent_size=64,
    )
    print(f"use_lora: {use_lora}")
    print(f"visual_und: {visual_und}")
    
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        if visual_und:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        if use_lora:
            pass
    
    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    
    # Device mapping
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    
    print(f"device_map: {device_map}")
    same_device_modules = [
        'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed'
    ]
    
    print(f"device_count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    if use_lora:
        pass
    else:
        model = load_checkpoint_and_dispatch(
            model, checkpoint=os.path.join(ema_model_path, "ema.safetensors"),
            device_map=device_map, offload_buffers=True, dtype=torch.bfloat16,
            force_hooks=True, offload_folder="/tmp/offload"
        )
    
    model = model.eval()
    print('Model loaded')

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer, vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids
    )
    return inferencer

def generate_image(sample_id, row, inferencer, output_dir):
    image_list = row['image_list']
    image_num = len(image_list)

    def resize_image_with_constraints(image):

        def make_divisible(value, stride=16):
            return max(stride, int(round(value / stride) * stride))
        
        w, h = image.size
        max_image_size = 1024
        min_image_size = 512
        scale = min(max_image_size / max(w, h), 1.0)
        scale = max(scale, min_image_size / min(w, h))
        new_w = make_divisible(w * scale)
        new_h = make_divisible(h * scale)
        if (new_w, new_h) != (w, h):
            image = image.resize((new_w, new_h), Image.LANCZOS)
            print(f'New image size is: {image.size}.')
            
        return image
    
    def _parse_steps(full_steps):
        import re
        pattern = r"Step\d+:\s*(.*?)(?=,\s*Step|$)"
        steps = re.split(pattern, full_steps)
        steps = [s.strip() for s in steps if s.strip()]
        return steps
    
    from PIL import Image
    if image_num != 4:
        return
    
    inference_hyper = dict(
        do_sample=True,
        text_temperature=0.3,
        cfg_text_scale = 4.0,
        cfg_img_scale = 2.0,
        cfg_interval = [0.0, 1.0],
        timestep_shift = 3.0,
        num_timesteps = 50,
        cfg_renorm_min = 0.0,
        cfg_renorm_type = "global",
    )
    output_image_shape = (1024, 1024)

    img_bytes = image_list[-1]
    imgs = Image.open(BytesIO(img_bytes))
    imgs = resize_image_with_constraints(imgs)
    imgs_path = os.path.join(output_dir, f"sample{sample_id}_gt.png") # save gt image
    imgs.save(imgs_path)
    prompt = row['instruction_list'][0]
    input_text = []
    
    TEXT_GENERATION_PROMPT = f'''
        Progressive Tutorial Generation Task 
        **Input Prompt**: "{prompt}"
        **Objective**: Generate a detailed 4-step tutorial showing the complete process.
        **Output Format** (Must follow exactly):
        ```
        StepX: [.......]
        ```
        **Requirements**:
        1. Each step builds upon the previous naturally
        2. Use clear, actionable language
        3. Describe visual elements that should appear
        4. Maintain consistency in style and terminology
        Now generate the tutorial for "{prompt}":
    '''
    
    input_text.append(TEXT_GENERATION_PROMPT)
    
    generate_step = image_num
    thinking = row['instruction_list'][1]
    output_thinking = inferencer.interleave_inference(
        input_lists = input_text, 
        understanding_output=True, # 先输出文本部分
        **inference_hyper
    )[0]
    
    print(output_thinking)
    step_texts = _parse_steps(output_thinking)
    step_texts = _parse_steps(thinking)
    
    for pos in range(generate_step): 
        current_input = []
        step_num = pos + 1
        current_step_text = step_texts[pos]
        GENERATION_PROMPT = f'''
            Multi-Frame Sequential Image Generation Task
            **Tutorial Context**: "{prompt}"
            **Core Objective**: Generate a sequence of {step_num} distinct, non-overlapping images that visually represent a step-by-step cooking tutorial.
            ---
            Scene description in this turn: {current_step_text}
            ---
            **CRITICAL OVERALL REQUIREMENTS**:
            1.  **Precise Adherence**: Every image must correspond **EXACTLY** to its step description. No creative interpretation that violates the text.
            2.  **Visual Consistency**: While the content of each frame is different, maintain a consistent **photorealistic style, lighting, and camera perspective** across all {step_num} frames to create a cohesive and professional tutorial feel.
            3.  **No Omissions, No Additions**: Generate ONLY the key elements described for each step. Avoid adding distracting background objects not mentioned in the tutorial context.
            Generate the {step_num}-frame sequence now:
        '''
        
        IMAGE_GENERATION_PROMPT = f'frame {pos+1} {step_texts[pos]}'
        current_input.append(GENERATION_PROMPT)
        current_input.append(IMAGE_GENERATION_PROMPT)
        new_img = inferencer.interleave_inference(
            input_lists=current_input, 
            image_shapes=output_image_shape,
            **inference_hyper
        )[0]
        
        new_img_path = os.path.join(output_dir, f"sample{sample_id}_gen_{pos+1}.png")
        new_img.save(new_img_path)
        print(f"save as path: {new_img_path}")

    result_data = {
        "sample_id": sample_id,
        "prompt": prompt,
        "text_description": output_thinking,
        "thinking": thinking,
        "gt_image": f"sample{sample_id}_gt.png"
    }
    
    import json
    json_path = os.path.join(output_dir, f"sample{sample_id}_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2) # 保存为json格式的文件
    print(f"JSON saved to: {json_path}")
    return result_data

def main(parquet_path, output_dir, inferencer):
    df = pd.read_parquet(parquet_path)
    Path(output_dir).mkdir(
        parents = True, 
        exist_ok = True
    )

    for idx, row in df.iterrows():
        generate_image(idx, row, inferencer, output_dir)

if __name__ == "__main__":
    args = parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inferencer = initialize_model_and_inferencer(
        model_path=args.model_path,
        ema_model_path=args.ema_model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        use_lora=args.use_lora,
        visual_und=args.visual_und,
        visual_gen=args.visual_gen
    )

    main(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        inferencer=inferencer
    )