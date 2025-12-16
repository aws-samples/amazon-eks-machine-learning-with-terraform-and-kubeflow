import json
import os
import time
import torch
import math
import numpy as np
import itertools
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import triton_python_backend_utils as pb_utils

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

_MODEL_ARGS_FILENAME = "model.json"


class TritonPythonModel:
    """Sentence Embeddings - Standalone implementation (too different for base class)"""
    
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        embeddings_config = pb_utils.get_output_config_by_name(self.model_config, "embeddings")
        self.embeddings_dtype = pb_utils.triton_string_to_numpy(embeddings_config["data_type"])
        self.example_text = 'The giant panda, sometimes called a panda bear, or simply panda, is a bear species endemic to China.'
        
        self._current_device = self._get_current_device()
        self._is_xla = xm is not None
        self._init_service()
        
        self.logger.log_info("Sentence Embeddings Model initialized")
    
    def _get_current_device(self) -> torch.device:
        """Get the appropriate device for inference"""
        if xm is not None:
            return xm.xla_device()
        elif torch.cuda.is_available():
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            current_device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(current_device)
            return current_device
        else:
            return torch.device("cpu")
    
    @staticmethod
    def min_power_of_2(n: int) -> int:
        return 2**math.ceil(math.log2(n))
    
    def _get_bucket_batch_size(self, n: int) -> int:
        assert n > 0, f"batch_size {n} is not > 0"
        n = self.min_power_of_2(n)
        for bs in self.bucket_batch_size:
            if bs >= n:
                return bs
        return self.max_batch_size
    
    def _get_bucket_seq_len(self, n: int) -> int:
        n = self.min_power_of_2(n)
        for seq_len in self.bucket_seq_len:
            if seq_len >= n:
                return seq_len
        return self.max_seq_len
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to get sentence embeddings"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _cls_pooling(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Use CLS token embedding as sentence embedding"""
        return token_embeddings[:, 0]
    
    def _bucket_batch_inference(self, inputs: dict) -> torch.Tensor:
        """Run inference and return embeddings"""
        with torch.no_grad():
            # Move inputs to device properly
            inputs = {k: v.to(self._current_device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True)
            
            # Get embeddings based on pooling strategy
            if self.pooling_strategy == "mean":
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            elif self.pooling_strategy == "cls":
                embeddings = self._cls_pooling(outputs.last_hidden_state)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
            # Normalize if required
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Synchronize XLA operations
            if self._is_xla:
                xm.mark_step()
            
            return embeddings.detach().cpu()
    
    def _run_inference(self, texts: list) -> list:
        """Run inference and return sentence embeddings"""
        start = time.time()
        pad_token_id = self.tokenizer.pad_token_id or 1
        input_batch_size = len(texts)
        
        assert input_batch_size <= self.max_batch_size, \
            f"input_batch_size: {input_batch_size} is > max_batch_size: {self.max_batch_size}"
        
        # Only pad batch size for XLA
        if self._is_xla:
            pad_batch_size = self._get_bucket_batch_size(input_batch_size)
            padded_texts = texts + [self.example_text] * (pad_batch_size - input_batch_size)
        else:
            padded_texts = texts
        
        # Tokenize
        inputs = self.tokenizer(
            padded_texts,
            padding="longest",
            truncation=True,
            return_tensors='pt',
            max_length=self.max_seq_len
        )
        
        input_seq_len = inputs['input_ids'].shape[-1]
        
        # Only pad sequence length for XLA
        if self._is_xla:
            pad_seq_len = self._get_bucket_seq_len(input_seq_len)
            padding = pad_seq_len - input_seq_len
            if padding > 0:
                inputs['input_ids'] = F.pad(inputs['input_ids'], (0, padding), 'constant', pad_token_id)
                inputs['attention_mask'] = F.pad(inputs['attention_mask'], (0, padding), 'constant', 0)
        
        # Run inference
        embeddings = self._bucket_batch_inference(inputs)
        
        # Extract only the original batch
        embeddings = embeddings[:input_batch_size]
        
        # Convert to list for Triton
        results = [embeddings[i].numpy().astype(self.embeddings_dtype).tolist() 
                   for i in range(input_batch_size)]
        
        int_time = time.time() - start
        self.logger.log_info(
            f"Model input_batch_size: {input_batch_size} input_seq_len: {input_seq_len}, "
            f"inference time: {int_time:.4f}s"
        )
        
        return results
    
    def _compile_model(self):
        """Compile model for target device"""
        if self._is_xla:
            permutations = list(itertools.product(self.bucket_batch_size, self.bucket_seq_len))
            self.logger.log_info(f"Compiling {len(permutations)} XLA model variants...")
            
            for batch_size, seq_len in permutations:
                self.logger.log_info(f"Compiling batch_size={batch_size}, seq_len={seq_len}")
                texts = [self.example_text] * batch_size
                inputs = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    return_tensors='pt',
                    max_length=seq_len
                )
                self._bucket_batch_inference(inputs)
            
            self.logger.log_info("XLA compilation complete")
        else:
            self.logger.log_info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
    
    def _init_service(self):
        """Initialize model service"""
        max_batch_size = int(self.model_config.get('max_batch_size', 8))
        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)
        
        assert not using_decoupled, \
            "Triton Server Python backend must not use decoupled model transaction policy"
        
        model_args_filepath = os.path.join(pb_utils.get_model_dir(), _MODEL_ARGS_FILENAME)
        assert os.path.isfile(model_args_filepath), \
            f"'{_MODEL_ARGS_FILENAME}' containing model args must be provided in '{pb_utils.get_model_dir()}'"
        
        with open(model_args_filepath) as file:
            properties = json.load(file)
        
        # Initialize bucket configurations
        self.bucket_batch_size = sorted(properties.get("bucket_batch_size", [1, 2, 4, 8]))
        self.bucket_seq_len = sorted(properties.get("bucket_seq_len", [32, 64, 128]))
        
        # Validate power of 2
        for bs in self.bucket_batch_size:
            assert (bs & (bs-1) == 0), f"bucket batch size {bs} is not power of 2"
        
        for bsl in self.bucket_seq_len:
            assert (bsl & (bsl-1) == 0), f"bucket seq len {bsl} is not power of 2"
        
        self.max_batch_size = max(self.bucket_batch_size)
        self.max_seq_len = max(self.bucket_seq_len)
        
        assert self.max_batch_size == max_batch_size, \
            f"Triton Server max_batch_size {max_batch_size} != model max_batch_size: {self.max_batch_size}"
        
        # Embedding-specific configuration
        self.pooling_strategy = properties.get("pooling_strategy", "mean")  # "mean" or "cls"
        self.normalize = properties.get("normalize", True)  # L2 normalization
        
        assert self.pooling_strategy in ["mean", "cls"], \
            f"pooling_strategy must be 'mean' or 'cls', got {self.pooling_strategy}"
        
        # Load model and tokenizer
        model_location = properties.get("model_id_or_path")
        self.logger.log_info(f"Loading model from {model_location}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_location)
        self.model = AutoModel.from_pretrained(model_location)  # Note: AutoModel, not task-specific
        self.model.eval()
        
        # Move to device and compile
        self.logger.log_info(f"Moving model to device and compiling...")
        path = os.getcwd()
        os.chdir("/tmp")
        
        self.model.to(self._current_device)
        self._compile_model()
        
        os.chdir(path)
        self.logger.log_info("Model initialization complete")
    
    def execute(self, requests):
        """Execute inference requests"""
        responses = []
        texts = []
        
        # Extract all text inputs from requests
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            inputs = input_tensor.as_numpy().tolist()
            assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
            
            # Handle bytes or string
            text_item = inputs[0][0]
            if isinstance(text_item, bytes):
                text_item = text_item.decode("utf-8")
            
            texts.append(text_item)
        
        # Run batch inference
        embeddings = self._run_inference(texts)
        
        # Create responses
        for result in embeddings:
            output_tensor = pb_utils.Tensor("embeddings", np.array(result, dtype=self.embeddings_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        assert len(responses) == len(requests), \
            f"num responses: {len(responses)} != num requests {len(requests)}"
        
        return responses
    
    def finalize(self):
        """Cleanup resources"""
        self.logger.log_info("Cleaning up...")
    
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Auto-complete Triton config for sentence embeddings"""
        inputs = [{"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]}]
        outputs = [{"name": "embeddings", "data_type": "TYPE_FP32", "dims": [-1]}]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_model_transaction_policy(dict(decoupled=False))
        auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config