import io
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class MakeAnythingIconUnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    
    def parse_row(self, row):
        # cooking tutorial
        data = self._init_data()
        
        prompt = row['instruction_list'][0]
        full_steps = row['instruction_list'][1]
        step_num = len(row['image_list'])
        step_texts = self._parse_steps(full_steps) 
        
        TEXT_GENERATION_PROMPT = f'''
            Progressive Tutorial Generation Task
            **Input Prompt**: "{prompt}"
            **Objective**: Generate a detailed {step_num} steps tutorial showing the complete process.
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
        
        data = self._add_text(data, TEXT_GENERATION_PROMPT, need_loss=False, enable_cfg=True)
        data = self._add_text(data, full_steps, need_loss=True, enable_cfg=True) # Text generation at first
        
        IMAGE_GENERATION_PROMPT = f'''
            Multi-Frame Sequential Image Generation Task
            **Tutorial Context**: "{prompt}"
            **Core Objective**: Generate a sequence of {step_num} distinct, non-overlapping images that visually represent a step-by-step cooking tutorial.
            ---
            Scene description in this turn: {step_texts[step_num-1]}
            ---
            **CRITICAL OVERALL REQUIREMENTS**:
            1.  **Precise Adherence**: Every image must correspond **EXACTLY** to its step description. No creative interpretation that violates the text.
            2.  **Visual Consistency**: While the content of each frame is different, maintain a consistent **photorealistic style, lighting, and camera perspective** across all {step_num} frames to create a cohesive and professional tutorial feel.
            3.  **No Omissions, No Additions**: Generate ONLY the key elements described for each step. Avoid adding distracting background objects not mentioned in the tutorial context.
            Generate the {step_num}-frame sequence now:
        '''
    
        data = self._add_text(data, IMAGE_GENERATION_PROMPT, need_loss=False, enable_cfg=True)
        
        for idx in range(step_num-1):
            # add preview images as context.
            data = self._add_text(data, f"<|turn_{idx+1}|>", need_loss=False, enable_cfg=True)
            data = self._add_text(data, f"{step_texts[idx]}", need_loss=False, enable_cfg=True)
            data = self._add_image(
                data, 
                pil_img2rgb(self.resize_image_with_constraints(Image.open(io.BytesIO(row["image_list"][idx])))),
                need_loss=False, 
                need_vae=True, 
                need_vit=True,
            )
            
        # add the last image for prediction.
        data = self._add_text(data, f"<|turn_{idx+1}|>", need_loss=False, enable_cfg=True)
        data = self._add_text(data, f"{step_texts[idx]}", need_loss=False, enable_cfg=True)
        data = self._add_image(
            data, 
            pil_img2rgb(self.resize_image_with_constraints(Image.open(io.BytesIO(row["image_list"][step_num-1])))),
            need_loss=True, 
            need_vae=False, 
            need_vit=False,
        )
        
        return data
    
    def resize_image_with_constraints(self, image):
        
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
        return image
    
    def _parse_steps(self, full_steps):
        import re
        
        pattern = r"Step\d+:\s*(.*?)(?=,\s*Step|$)" 
        matches = re.findall(pattern, full_steps)
        steps = [s.strip() for s in matches]
        return steps