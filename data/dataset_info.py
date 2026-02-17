# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, Interleaved_Input, MakeAnythingIconUnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'interleaved_input':Interleaved_Input,
    'interleaved_output':MakeAnythingIconUnifiedEditIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # 可以选择性加入interleaved output section当中的分步操作进行sft对齐，比如做菜教程，我们拆解成不同的X步，每次只训练一步的T2I
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'interleaved_input': {
        'seedxedit_multi': {
            'data_dir': 'loom_data_path/interleaved_input/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'loom_data_path/interleaved_output/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
        },
    },
    'interleaved_output': {
        'seedxedit_multi': {
            'data_dir': 'loom_data_path/interleaved_output/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'loom_data_path/interleaved_output/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
}