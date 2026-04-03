#!/usr/bin/env python3
"""
Test script for auto-interleaved generation functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image

import sys
sys.path.append(".")

from auto_interleaved_inference import AutoInterleavedInferencer


class TestAutoInterleavedInferencer(unittest.TestCase):
    """Test cases for AutoInterleavedInferencer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model components
        self.mock_model = MagicMock()
        self.mock_vae_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_vae_transform = MagicMock()
        self.mock_vit_transform = MagicMock()
        
        # Set up token IDs
        self.new_token_ids = {
            'start_of_image': 151652,
            'end_of_image': 151653,
            'eos_token_id': 151645,
            'bos_token_id': 151644
        }
        
        # Configure tokenizer mock
        self.mock_tokenizer.decode.side_effect = self._mock_decode
        self.mock_tokenizer.convert_tokens_to_ids.side_effect = self._mock_token_to_id
        
        # Create inferencer instance
        self.inferencer = AutoInterleavedInferencer(
            model=self.mock_model,
            vae_model=self.mock_vae_model,
            tokenizer=self.mock_tokenizer,
            vae_transform=self.mock_vae_transform,
            vit_transform=self.mock_vit_transform,
            new_token_ids=self.new_token_ids
        )
        
        # Set device and dtype
        self.inferencer.device = torch.device('cpu')
        self.inferencer.dtype = torch.float32
    
    def _mock_decode(self, token_ids):
        """Mock tokenizer decode function."""
        if isinstance(token_ids, list):
            # Map special token IDs to their string representations
            result = []
            for tid in token_ids:
                if tid == self.new_token_ids['start_of_image']:
                    result.append('<|vision_start|>')
                elif tid == self.new_token_ids['end_of_image']:
                    result.append('<|vision_end|>')
                elif tid == self.new_token_ids['eos_token_id']:
                    result.append('<|im_end|>')
                else:
                    result.append(f'token_{tid}')
            return ' '.join(result)
        return f'token_{token_ids}'
    
    def _mock_token_to_id(self, token_str):
        """Mock tokenizer token to ID conversion."""
        token_map = {
            '<|vision_start|>': self.new_token_ids['start_of_image'],
            '<|vision_end|>': self.new_token_ids['end_of_image'],
            '<|im_end|>': self.new_token_ids['eos_token_id'],
            '<|im_start|>': self.new_token_ids['bos_token_id']
        }
        return token_map.get(token_str, 1000)  # Default token ID
    
    def test_init(self):
        """Test proper initialization of AutoInterleavedInferencer."""
        self.assertEqual(self.inferencer.vision_start_token_id, 151652)
        self.assertEqual(self.inferencer.vision_end_token_id, 151653)
        self.assertEqual(self.inferencer.eos_token_id, 151645)
    
    def test_generate_next_token_shape(self):
        """Test that _generate_next_token returns proper shape."""
        # Mock context
        mock_context = {
            'past_key_values': MagicMock(),
            'kv_lens': [torch.tensor([10])],
            'ropes': [torch.tensor([0.1])]
        }
        
        # Mock model methods
        self.mock_model.prepare_start_tokens.return_value = {
            'packed_start_tokens': torch.tensor([1]),
            'packed_query_position_ids': torch.tensor([0]),
            'packed_key_value_indexes': torch.tensor([0])
        }
        
        # Mock language model components
        mock_output = MagicMock()
        mock_output.packed_query_sequence = torch.randn(1, 1, 768)
        
        self.mock_model.language_model.model.embed_tokens.return_value = torch.randn(1, 768)
        self.mock_model.language_model.forward_inference.return_value = mock_output
        self.mock_model.language_model.lm_head.return_value = torch.randn(1, 50000)
        self.mock_model.use_moe = False
        
        # Call method
        with patch('torch.cumsum', return_value=torch.tensor([10])):
            result = self.inferencer._generate_next_token(mock_context)
        
        # Verify result
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dim(), 0)  # Should be a scalar
    
    def test_vision_token_detection(self):
        """Test detection of vision start tokens in generated sequence."""
        # This tests the logic for detecting when to switch to image generation
        prompt = "Generate an image of a cat"
        
        # Mock the generation to return vision start token
        with patch.object(self.inferencer, '_generate_next_token') as mock_gen:
            # First return some text tokens, then vision start token
            mock_gen.side_effect = [
                torch.tensor(100),  # Regular token
                torch.tensor(101),  # Regular token
                torch.tensor(self.new_token_ids['start_of_image']),  # Vision start
            ]
            
            with patch.object(self.inferencer, 'gen_image') as mock_gen_image:
                mock_gen_image.return_value = Image.new('RGB', (64, 64))
                
                with patch.object(self.inferencer, 'update_context_text') as mock_update_text:
                    mock_update_text.return_value = {'past_key_values': MagicMock(), 'kv_lens': [10], 'ropes': [0.1]}
                    
                    with patch.object(self.inferencer, 'update_context_image') as mock_update_image:
                        mock_update_image.return_value = {'past_key_values': MagicMock(), 'kv_lens': [10], 'ropes': [0.1]}
                        
                        # Run generation with max 1 block
                        outputs = self.inferencer.auto_interleaved_generation(
                            prompt, 
                            max_interleaved_blocks=1,
                            max_text_length=10
                        )
            
            # Should have generated text and then an image
            self.assertEqual(len(outputs), 2)
            self.assertIsInstance(outputs[0], str)
            self.assertIsInstance(outputs[1], Image.Image)
    
    def test_eos_token_handling(self):
        """Test proper handling of EOS token."""
        prompt = "Test prompt"
        
        # Mock generation to return EOS token immediately
        with patch.object(self.inferencer, '_generate_next_token') as mock_gen:
            mock_gen.return_value = torch.tensor(self.new_token_ids['eos_token_id'])
            
            with patch.object(self.inferencer, 'update_context_text') as mock_update:
                mock_update.return_value = {'past_key_values': MagicMock(), 'kv_lens': [10], 'ropes': [0.1]}
                
                outputs = self.inferencer.auto_interleaved_generation(prompt)
        
        # Should return empty list since EOS was hit immediately
        self.assertEqual(len(outputs), 0)
    
    def test_max_blocks_limit(self):
        """Test that generation respects max_interleaved_blocks limit."""
        prompt = "Test prompt"
        max_blocks = 2
        
        # Mock to generate regular tokens (no special tokens)
        with patch.object(self.inferencer, '_generate_next_token') as mock_gen:
            mock_gen.return_value = torch.tensor(100)  # Regular token
            
            with patch.object(self.inferencer, 'update_context_text') as mock_update:
                mock_update.return_value = {'past_key_values': MagicMock(), 'kv_lens': [10], 'ropes': [0.1]}
                
                outputs = self.inferencer.auto_interleaved_generation(
                    prompt,
                    max_interleaved_blocks=max_blocks,
                    max_text_length=5  # Small limit for testing
                )
        
        # Should have at most max_blocks outputs
        self.assertLessEqual(len(outputs), max_blocks)
    
    def test_think_mode(self):
        """Test that think mode adds appropriate system prompt."""
        prompt = "Test prompt"
        
        with patch.object(self.inferencer, 'update_context_text') as mock_update:
            mock_update.return_value = {'past_key_values': MagicMock(), 'kv_lens': [10], 'ropes': [0.1]}
            
            with patch.object(self.inferencer, '_generate_next_token') as mock_gen:
                # Return EOS immediately to end generation
                mock_gen.return_value = torch.tensor(self.new_token_ids['eos_token_id'])
                
                outputs = self.inferencer.auto_interleaved_generation(
                    prompt,
                    think=True
                )
        
        # Check that system prompt was added
        calls = mock_update.call_args_list
        system_prompt_added = any(
            'think about the planning process' in str(call)
            for call in calls
        )
        self.assertTrue(system_prompt_added)


class TestIntegration(unittest.TestCase):
    """Integration tests requiring more setup."""
    
    @unittest.skip("Requires actual model weights")
    def test_full_generation_pipeline(self):
        """Test full generation pipeline with real model."""
        # This would require actual model weights and setup
        pass


if __name__ == '__main__':
    unittest.main()