import io
from PIL import Image, ImageFile, PngImagePlugin
from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class InterleavedInputUnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        # For Interleaved Input Task.
        image_num = len(row["image_list"])
        end_idx = image_num - 1 
        
        data = self._init_data()
        for idx in range(end_idx): 
            # Add all reference images as conditions.
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                need_loss = False, 
                need_vae = True, 
                need_vit = True,
            )
            
        generation = row['instruction_list'][0]
        data = self._add_text(data, generation, need_loss=False, enable_cfg=True)
        
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
            need_loss = True, 
            need_vae = False, 
            need_vit = False,
        )

        return data   