# Loom Inference code

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
    parser.add_argument('--use_lora', type=bool, help='Whether to use LoRA')
    parser.add_argument('--visual_gen', type=bool, help='Enable visual generation')
    parser.add_argument('--visual_und', type=bool, help='Enable visual understanding')
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
    instruction = row['instruction_list'][0]
    image_list = row['image_list']

    if len(image_list) > 1:
        gt_img = Image.open(BytesIO(image_list[-1]))
        gt_path = os.path.join(output_dir, f"sample{sample_id}_gt.jpg")
        gt_img.save(gt_path)
        print(f"Saved ground truth: {gt_path}")

    inference_hyper = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )
    
    input_list = []
    for _, img_bytes in enumerate(image_list[:-1]):
        img = Image.open(BytesIO(img_bytes))
        input_list.append(img) # concat every image in one seq and generate the final image.
    
    input_list.append(instruction)
    output_dict = inferencer.interleave_inference(input_lists=input_list, **inference_hyper)
    generated_img = output_dict[0]
    
    gen_path = os.path.join(output_dir, f"sample{sample_id}_gen.jpg")
    generated_img.save(gen_path)
    print(f"Saved generated image: {gen_path}")
    
def main(parquet_path, end_id, output_dir, inferencer):
    df = pd.read_parquet(parquet_path)
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    for idx, row in df.iterrows():
        if idx <= end_id:
            # you can setting the batch generation.
            generate_image(idx, row, inferencer, output_dir)

if __name__ == "__main__":
    # official inference code for Loom's Interleaved Input Tasks.
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
        model_path=args.model_path, ema_model_path=args.ema_model_path, 
        max_mem_per_gpu=args.max_mem_per_gpu, use_lora=args.use_lora, visual_und=args.visual_und, visual_gen=args.visual_gen
    )
    
    main(parquet_path=args.parquet_path, end_id=args.end_id, output_dir=args.output_dir, inferencer=inferencer,)
    