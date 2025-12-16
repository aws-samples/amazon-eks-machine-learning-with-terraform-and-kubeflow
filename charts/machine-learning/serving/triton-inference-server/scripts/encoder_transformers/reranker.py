import numpy as np
from transformers import AutoModelForSequenceClassification
import triton_python_backend_utils as pb_utils
from encoder_base import EncoderBaseModel
import torch
import torch.nn.functional as F
import time
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

class TritonPythonModel(EncoderBaseModel):
    """Reranker for ranking/reranking - processes query with multiple documents efficiently"""
    
    def _init_output_config(self):
        """Initialize output tensor configuration for reranker"""
        scores_config = pb_utils.get_output_config_by_name(self.model_config, "scores")
        self.scores_dtype = pb_utils.triton_string_to_numpy(scores_config["data_type"])
        
        # Reranker specific example
        self.example_pair = ['what is panda?', 
                            'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
    
    def _load_model(self, model_location):
        """Load reranker model (sequence classification for ranking)"""
        self.logger.log_info(f"Loading Reranker from {model_location}")
        return AutoModelForSequenceClassification.from_pretrained(model_location)
    
    def _bucket_batch_inference_pairs(self, inputs: dict) -> torch.Tensor:
        """Run inference on query-document pairs
        
        Args:
            inputs: Tokenized inputs dict with input_ids, attention_mask, token_type_ids
            
        Returns:
            Tensor of relevance scores
        """
        with torch.no_grad():
            # Move inputs to device
            inputs = {k: v.to(self._current_device) for k, v in inputs.items()}
            
            # Get scores
            logits = self.model(**inputs, return_dict=True).logits
            
            # Synchronize XLA operations
            if self._is_xla:
                xm.mark_step()
            
            # Get relevance scores (squeeze if single output class)
            if logits.shape[-1] == 1:
                scores = logits.view(-1).float()
            else:
                # Multi-class: use softmax on positive class or just first class
                scores = torch.softmax(logits, dim=-1)[:, 0]
            
            return scores.detach().cpu()
    
    def _run_inference_pairs(self, pairs: list) -> list:
        """Run inference on query-document pairs with XLA bucketing
        
        Args:
            pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores
        """
        start = time.time()
        pad_token_id = self.tokenizer.pad_token_id or 1
        batch_size = len(pairs)
        
        assert batch_size <= self.max_batch_size, \
            f"batch_size: {batch_size} is > max_batch_size: {self.max_batch_size}"
        
        # Only pad batch size for XLA
        if self._is_xla:
            bucket_batch_size = self._get_bucket_batch_size(batch_size)
            padded_pairs = pairs + [self.example_pair for _ in range(bucket_batch_size - batch_size)]
        else:
            padded_pairs = pairs
        
        # Tokenize pairs - tokenizer handles [query, document] pairs automatically
        print(f"Tokenizing {padded_pairs} pairs")
        inputs = self.tokenizer(
            padded_pairs, 
            padding="longest",  # First pad to longest in batch
            truncation=True, 
            return_tensors='pt', 
            max_length=self.max_seq_len
        )
        
        input_seq_len = inputs['input_ids'].shape[-1]
        
        # Only pad sequence length for XLA (bucket to power of 2)
        if self._is_xla:
            pad_seq_len = self._get_bucket_seq_len(input_seq_len)
            padding = pad_seq_len - input_seq_len
            if padding > 0:
                inputs['input_ids'] = F.pad(inputs['input_ids'], (0, padding), 'constant', pad_token_id)
                inputs['attention_mask'] = F.pad(inputs['attention_mask'], (0, padding), 'constant', 0)
                # Also pad token_type_ids if present
                if 'token_type_ids' in inputs:
                    inputs['token_type_ids'] = F.pad(inputs['token_type_ids'], (0, padding), 'constant', 0)
        
        # Run inference
        scores = self._bucket_batch_inference_pairs(inputs)
        
        # Extract only original batch
        scores = scores[:batch_size]
        
        # Convert to list
        results = scores.numpy().astype(self.scores_dtype).tolist()
        
        int_time = time.time() - start
        self.logger.log_info(
            f"Reranker batch_size: {batch_size} input_seq_len: {input_seq_len}, "
            f"inference time: {int_time:.4f}s"
        )
        
        return results
    
    def _compile_model(self):
        """Override compilation to use pair-based inference with bucketing"""
        if self._is_xla:
            import itertools
            
            # Compile all bucket combinations for XLA
            permutations = list(itertools.product(self.bucket_batch_size, self.bucket_seq_len))
            self.logger.log_info(f"Compiling {len(permutations)} reranker XLA variants...")
            
            for batch_size, seq_len in permutations:
                self.logger.log_info(f"Compiling batch_size={batch_size}, seq_len={seq_len}")
                
                # Create example pairs
                pairs = [self.example_pair for _ in range(batch_size)]
                
                # Tokenize with exact padding to seq_len
                inputs = self.tokenizer(
                    pairs,
                    padding="max_length",
                    truncation=True,
                    return_tensors='pt',
                    max_length=seq_len
                )
                
                # Run inference to compile this shape
                self._bucket_batch_inference_pairs(inputs)
            
            self.logger.log_info("XLA compilation complete")
        else:
            # Use torch.compile for non-XLA devices
            self.logger.log_info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
    
    def execute(self, requests):
        """Execute reranker inference - one query with multiple documents per request"""
        responses = []
        
        for request in requests:
            try:
                # Get query and list of texts
                query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
                texts_tensor = pb_utils.get_input_tensor_by_name(request, "texts")
                
                # Extract and decode query
                query_array = query_tensor.as_numpy()
                
                # Handle different query shapes: (1,), (1, 1), etc.
                query_flat = query_array.flatten()
                if len(query_flat) == 0:
                    raise ValueError("Query is empty")
                
                query_item = query_flat[0]
                query_str = query_item.decode("utf-8") if isinstance(query_item, bytes) else str(query_item)
                
                # Extract and decode texts
                texts_array = texts_tensor.as_numpy()
                texts_flat = texts_array.flatten()
                
                if len(texts_flat) == 0:
                    raise ValueError("Texts array is empty")
                
                texts_list = []
                for text_item in texts_flat:
                    text_str = text_item.decode("utf-8") if isinstance(text_item, bytes) else str(text_item)
                    texts_list.append(text_str)
                
                # Create pairs
                pairs = [[query_str, text_str] for text_str in texts_list]
                
                self.logger.log_info(
                    f"Processing query: '{query_str[:50]}...' with {len(pairs)} documents"
                )
                
                # Limit to max batch size if needed
                if len(pairs) > self.max_batch_size:
                    self.logger.log_warning(
                        f"Number of documents ({len(pairs)}) exceeds max_batch_size ({self.max_batch_size}). "
                        f"Truncating to {self.max_batch_size} documents."
                    )
                    pairs = pairs[:self.max_batch_size]
                
                # Run inference on all pairs
                scores = self._run_inference_pairs(pairs)
                
                # Create response
                out_tensor = pb_utils.Tensor("scores", np.array(scores, dtype=self.scores_dtype))
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)
                
            except Exception as e:
                self.logger.log_error(f"Error processing request: {str(e)}")
                # Return error response
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Processing failed: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    # Note: We don't use the base class methods _process_outputs and _create_response
    # because reranker has a different workflow (pairs instead of single texts)
    def _process_outputs(self, model_output, original_lengths):
        """Not used for reranker - kept for base class compatibility"""
        raise NotImplementedError("Reranker uses _run_inference_pairs instead")
    
    def _create_response(self, result):
        """Not used for reranker - kept for base class compatibility"""
        raise NotImplementedError("Reranker creates responses in execute()")
    
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Auto-complete Triton config for reranker"""
        inputs = [
            {"name": "query", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "texts", "data_type": "TYPE_STRING", "dims": [-1]}
        ]
        outputs = [{"name": "scores", "data_type": "TYPE_FP32", "dims": [-1]}]

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