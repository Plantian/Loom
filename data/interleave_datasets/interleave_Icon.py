import io
import pyarrow.parquet as pq

from PIL import Image, ImageFile, PngImagePlugin
from ..distributed_iterable_dataset import DistributedIterableDataset
from ..parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from ..data_utils_special import pil_img2rgb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class InterleavedBaseIterableDataset(DistributedIterableDataset):

    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        return data


class ParquetStandardIterableDataset(DistributedIterableDataset):

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data, parquet_info,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        vit_transform: input transform for vit model.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data, parquet_info)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data, parquet_info):
        row_groups = []
        for data_dir, num_data_path in zip(data_dir_list, num_used_data):
            data_paths = get_parquet_data_paths([data_dir], [num_data_path])
            for data_path in data_paths:
                if data_path in parquet_info.keys():
                    num_row_groups = parquet_info[data_path]['num_row_groups']
                    for rg_idx in range(num_row_groups):
                        row_groups.append((data_path, rg_idx))
        return row_groups

    def parse_row(self, row):
        raise NotImplementedError

    def __iter__(self):
        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            global_row_group_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            global_row_group_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at global_rg#{global_row_group_start_id}, row#{row_start_id}"
        )

        while True:
            file_paths_per_worker_ = file_paths_per_worker[global_row_group_start_id:]
            for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(
                file_paths_per_worker_, start=global_row_group_start_id
            ):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    try:
                        fr = pq.ParquetFile(f)
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]
                    except Exception as e:
                        print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                        continue

                    for row_idx, row in df.iterrows():
                        try:
                            data = self.parse_row(row)
                            if len(data) == 0:
                                continue
                            data['data_indexes'] = {
                                "data_indexes": [global_row_group_idx, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                        except Exception as e:
                            print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                            continue
                        yield data

                    row_start_id = 0
            global_row_group_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class MakeAnythingIconUnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    
    def parse_row(self, row):
        # cooking tutorial
        data = self._init_data()
        
        prompt = row['instruction_list'][0]
        full_steps = row['instruction_list'][1]
        step_num = len(row['image_list'])
        step_texts = self._parse_steps(full_steps) 
        
        assert len(step_texts) == step_num, "[Warning]: image num != text step"
        
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