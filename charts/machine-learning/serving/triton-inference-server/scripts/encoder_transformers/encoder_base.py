import json
import os
import time
import torch
import math
import itertools
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils
from abc import ABC, abstractmethod

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

_MODEL_ARGS_FILENAME = "model.json"

class EncoderBaseModel(ABC):
    """Base class for all encoder transformer tasks optimized for XLA Bucketing"""
    
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        self.example_text = 'The giant panda, sometimes called a panda bear, or simply panda, is a bear species endemic to China.'
        
        self._current_device = self._get_current_device()
        self._is_xla = xm is not None
        
        self._init_output_config()
        self._init_service()
        self.logger.log_info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def _init_output_config(self):
        pass

    @abstractmethod
    def _load_model(self, model_location):
        pass

    @abstractmethod
    def _process_outputs(self, model_output, attention_mask):
        pass

    @abstractmethod
    def _create_response(self, request_results):
        pass

    def _get_current_device(self) -> torch.device:
        if xm is not None:
            return xm.xla_device()
        elif torch.cuda.is_available():
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            current_device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(current_device)
            return current_device
        return torch.device("cpu")

    def _get_bucket_batch_size(self, n: int) -> int:
        for bs in self.bucket_batch_size:
            if bs >= n:
                return bs
        return self.max_batch_size

    def _get_bucket_seq_len(self, n: int) -> int:
        for seq_len in self.bucket_seq_len:
            if seq_len >= n:
                return seq_len
        return self.max_seq_len

    def _bucket_batch_inference(self, inputs: dict) -> object:
        with torch.no_grad():
            inputs = {k: v.to(self._current_device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True)
            if self._is_xla:
                xm.mark_step()
            return outputs

    def _run_inference(self, texts: list) -> list:
        start = time.time()
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_batch_size = len(texts)
        
        # 1. Handle Batch Bucketing
        if self._is_xla:
            pad_batch_size = self._get_bucket_batch_size(input_batch_size)
            # Ensure the list of strings matches the compiled batch size
            padded_texts = texts + [self.example_text] * (pad_batch_size - input_batch_size)
        else:
            padded_texts = texts

        # 2. Tokenization
        inputs = self.tokenizer(
            padded_texts,
            padding="longest",
            truncation=True,
            return_tensors='pt',
            max_length=self.max_seq_len
        )
        
        current_seq_len = inputs['input_ids'].shape[-1]
        
        # 3. Handle Sequence Length Bucketing for XLA
        if self._is_xla:
            target_seq_len = self._get_bucket_seq_len(current_seq_len)
            delta = target_seq_len - current_seq_len
            
            if delta > 0:
                # Pad ALL tensors present in inputs (input_ids, attention_mask, token_type_ids)
                for key in inputs.keys():
                    pad_val = pad_token_id if key == 'input_ids' else 0
                    inputs[key] = F.pad(inputs[key], (0, delta), value=pad_val)
        
        # 4. Inference
        outputs = self._bucket_batch_inference(inputs)
        
        results = self._process_outputs(outputs, inputs['attention_mask'])
        results = results[:input_batch_size]
        
        self.logger.log_info(f"Batch: {input_batch_size}, Time: {time.time() - start:.4f}s")
        return results

    def _init_service(self):
        model_args_filepath = os.path.join(pb_utils.get_model_dir(), _MODEL_ARGS_FILENAME)
        with open(model_args_filepath) as file:
            properties = json.load(file)
        
        self.bucket_batch_size = sorted(properties.get("bucket_batch_size", [1, 2, 4, 8])) if self._is_xla else [1]
        self.bucket_seq_len = sorted(properties.get("bucket_seq_len", [32, 64, 128])) if self._is_xla else [128]
        
        self.max_batch_size = max(self.bucket_batch_size)
        self.max_seq_len = max(self.bucket_seq_len)
        
        model_location = properties.get("model_id_or_path")
        self.tokenizer = AutoTokenizer.from_pretrained(model_location)
        self.model = self._load_model(model_location)
        self.model.to(self._current_device).eval()
        
        self._compile_model()

    def _compile_model(self):
        """Pre-compiles the model for every bucket permutation to avoid JIT at request time"""
        if self._is_xla:
            perms = list(itertools.product(self.bucket_batch_size, self.bucket_seq_len))
            self.logger.log_info(f"Starting XLA Compilation for {len(perms)} shapes...")
            for bs, sl in perms:
                texts = [self.example_text] * bs
                inputs = self.tokenizer(
                    texts, 
                    padding="max_length", # Static padding for compilation
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=sl
                )
                self._bucket_batch_inference(inputs)
            self.logger.log_info("XLA Compilation complete.")
        else:
            self.model = torch.compile(self.model)

    def execute(self, requests):
        responses = []
        all_texts = []
        request_item_counts = []
        
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            input_data = input_tensor.as_numpy()
            
            count = input_data.shape[0]
            request_item_counts.append(count)
            
            for i in range(count):
                text_item = input_data[i][0]
                if isinstance(text_item, bytes):
                    text_item = text_item.decode("utf-8")
                all_texts.append(str(text_item))
        
        # Batch execution
        all_results = self._run_inference(all_texts)
        
        # Split results back into individual requests
        curr = 0
        for count in request_item_counts:
            request_slice = all_results[curr : curr + count]
            curr += count
            responses.append(self._create_response(request_slice))
            
        return responses

    def finalize(self):
        self.logger.log_info("Cleaning up...")