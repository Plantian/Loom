import io
import re
from PIL import Image, ImageFile, PngImagePlugin
from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils_special import pil_img2rgb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class InterleavedInputUnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        instruction = row["instruction_list"][0]
        image_list = [pil_img2rgb(Image.open(io.BytesIO(img))) for img in row["image_list"]]
        elements = self.change_format(instruction)

        data = self._init_data()
        used_image_indices = set()

        for item in elements:
            if item['type'] == 'text':
                data = self._add_text(data, item['text'], need_loss=False)
            elif item['type'] == 'image':
                idx = item['index']
                if idx >= len(image_list):
                    print(f"[warning] Referenced <image {idx+1}> exceeds available images.")
                    continue
                used_image_indices.add(idx)
                is_last_image = idx == len(image_list) - 1
                data = self._add_image(
                    data,
                    image_list[idx],
                    need_loss=is_last_image,
                    need_vae=not is_last_image,
                    need_vit=not is_last_image,
                    enable_cfg=True,
                )
                
        last_idx = len(image_list) - 1
        if last_idx not in used_image_indices:
            data = self._add_image(
                data, 
                image_list[last_idx],
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
                enable_cfg=True
            )

        return data

    def change_format(self, text):
        pattern = re.compile(r'<image(?:[_\s]+)(\d+)>')
        elements = []
        last_idx = 0
        
        for match in pattern.finditer(text):
            start, end = match.span()
            image_idx = int(match.group(1)) - 1 

            if start > last_idx:
                part = text[last_idx:start].strip()
                if part:
                    elements.append({
                        'type': 'text', 
                        'has_loss': 0, 
                        'text': part
                    })
            elements.append({
                'type': 'image', 
                'index': image_idx
            })
            last_idx = end

        if last_idx < len(text):
            tail = text[last_idx:].strip()
            if tail:
                elements.append({
                    'type': 'text', 
                    'has_loss': 0, 
                    'text': tail
                })

        return elements