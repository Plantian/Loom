import os
import sys
import argparse
import random
import numpy as np
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from auto_interleaved_inference import AutoInterleavedInferencer
from data.transforms import ImageTransform
from data.data_utils_special import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae


def setup_model(model_path: str, max_mem_per_gpu: str = "80GiB"):
    """Load and setup the Bagel model with proper device mapping."""
    
    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    
    # Bagel config
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )
    
    # Initialize model with empty weights
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    # Setup device map
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    
    # Ensure same device for related modules
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
    
    # Load checkpoint
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    
    model = model.eval()
    print('Model loaded successfully')
    
    return model, vae_model


def main():
    parser = argparse.ArgumentParser(description="Auto-interleaved text and image generation with Bagel")
    parser.add_argument("--model-path", type=str, default="/checkpoint/dream_v2/transfusion/BAGEL-7B-MoT",
                        help="Path to the Bagel model checkpoint")
    parser.add_argument("--prompt", type=str, 
                        default="Create a step-by-step tutorial on how to make a paper airplane. Include images for each step.",
                        help="Input prompt for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-blocks", type=int, default=5, 
                        help="Maximum number of text/image blocks to generate")
    parser.add_argument("--max-text-length", type=int, default=200,
                        help="Maximum tokens per text block")
    parser.add_argument("--image-size", type=int, default=1024, help="Generated image size")
    parser.add_argument("--output-dir", type=str, default="./outputs", 
                        help="Directory to save generated outputs")
    parser.add_argument("--think", action="store_true", 
                        help="Enable thinking/planning before generation")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load model
    model, vae_model = setup_model(args.model_path)
    
    # Setup tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    # Setup image transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    
    # Create inferencer
    inferencer = AutoInterleavedInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )
    
    # Generation parameters
    gen_params = dict(
        think=args.think,
        max_text_length=args.max_text_length,
        max_interleaved_blocks=args.max_blocks,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shape=(args.image_size, args.image_size)
    )
    
    print(f"\\nGenerating with prompt: {args.prompt}")
    print("-" * 80)
    
    # Run auto-interleaved generation
    outputs = inferencer.auto_interleaved_generation(args.prompt, **gen_params)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save and display results
    print("\\nGenerated outputs:")
    print("-" * 80)
    
    text_blocks = []
    image_count = 0
    
    for i, output in enumerate(outputs):
        if isinstance(output, str):
            print(f"\\n[Text Block {len(text_blocks) + 1}]:")
            print(output)
            text_blocks.append(output)
        elif isinstance(output, Image.Image):
            image_count += 1
            image_path = os.path.join(args.output_dir, f"generated_image_{image_count}.png")
            output.save(image_path)
            print(f"\\n[Image {image_count}]: Saved to {image_path}")
    
    # Save full conversation
    with open(os.path.join(args.output_dir, "conversation.txt"), "w") as f:
        f.write(f"Prompt: {args.prompt}\\n")
        f.write("-" * 80 + "\\n\\n")
        
        text_idx = 0
        img_idx = 0
        for output in outputs:
            if isinstance(output, str):
                f.write(f"[Text Block {text_idx + 1}]:\\n{output}\\n\\n")
                text_idx += 1
            else:
                img_idx += 1
                f.write(f"[Image {img_idx}]: generated_image_{img_idx}.png\\n\\n")
    
    print(f"\\nGeneration complete! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()