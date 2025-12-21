import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
import triton_python_backend_utils as pb_utils
from encoder_base import EncoderBaseModel
import itertools

class TritonPythonModel(EncoderBaseModel):
    """
    Reranker implementation: 1 Query vs N Documents.
    Optimized for single-request execution (No Dynamic Batching).
    """
    
    def _init_output_config(self):
        """Initialize output configuration for reranker scores"""
        scores_config = pb_utils.get_output_config_by_name(self.model_config, "scores")
        self.scores_dtype = pb_utils.triton_string_to_numpy(scores_config["data_type"])
        
        # Example pair used for XLA compilation and internal padding
        self.example_pair = ["What is a panda?", "The giant panda is a bear endemic to China."]

    def _load_model(self, model_location):
        """Load reranker model (Cross-Encoder)"""
        self.logger.log_info(f"Loading Reranker from {model_location}")
        return AutoModelForSequenceClassification.from_pretrained(model_location)

    def _process_outputs(self, model_output, attention_mask):
        """Extract scores from classification logits"""
        logits = model_output.logits.detach().cpu()
        
        # Handle different Cross-Encoder output heads
        if logits.shape[-1] == 1:
            # Regression/Binary head: Shape [Batch]
            scores = logits.view(-1).float()
        else:
            # Classification head: Use softmax and take the positive class score
            # Usually index 1 (is_relevant) or the last index
            scores = torch.softmax(logits, dim=-1)[:, -1]

        return scores.numpy().astype(self.scores_dtype).tolist()

    def _create_response(self, request_results):
        """Create a single response for the N document scores"""
        # request_results is a list of floats for the documents in the request
        out_array = np.array(request_results, dtype=self.scores_dtype)
        out_tensor = pb_utils.Tensor("scores", out_array)
        return pb_utils.InferenceResponse(output_tensors=[out_tensor])

    def _compile_model(self):
        """Override to ensure Reranker compiles with [Query, Doc] pairs"""
        if self._is_xla:
            perms = list(itertools.product(self.bucket_batch_size, self.bucket_seq_len))
            self.logger.log_info(f"Starting Reranker XLA Compilation for {len(perms)} shapes...")
            for bs, sl in perms:
                # CRITICAL: Use example_pair to trigger token_type_ids generation
                texts = [self.example_pair] * bs
                inputs = self.tokenizer(
                    texts, 
                    padding="max_length",
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=sl
                )
                self._bucket_batch_inference(inputs)
            self.logger.log_info("Reranker XLA Compilation complete.")
        else:
            super()._compile_model()

    def execute(self, requests):
        """
        Execute reranker inference. 
        Enforces one request at a time (No Dynamic Batching logic).
        """
        responses = []
        
        # Even without dynamic batching, Triton passes a list. 
        # We iterate, but typically len(requests) will be 1.
        for request in requests:
            try:
                # 1. Extract Query (Single string)
                query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
                query_data = query_tensor.as_numpy().flatten()
                query_str = query_data[0].decode("utf-8") if isinstance(query_data[0], bytes) else str(query_data[0])
                
                # 2. Extract Texts (Multiple strings to rank)
                texts_tensor = pb_utils.get_input_tensor_by_name(request, "texts")
                texts_data = texts_tensor.as_numpy().flatten()
                
                # 3. Create Pairs: [Query, Doc1], [Query, Doc2]...
                pairs = []
                for text_item in texts_data:
                    text_str = text_item.decode("utf-8") if isinstance(text_item, bytes) else str(text_item)
                    pairs.append([query_str, text_str])
                
                # 4. Run inference on the document batch
                # Reuses the logic from our overridden _run_inference
                all_scores = self._run_inference(pairs)
                
                # 5. Create Response
                responses.append(self._create_response(all_scores))

            except Exception as e:
                self.logger.log_error(f"Error in Reranker execute: {str(e)}")
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Reranker failed: {str(e)}")
                ))
            
        return responses

    def _run_inference(self, pairs: list) -> list:
        """Optimized One-Pass Reranker inference for XLA"""
        input_batch_size = len(pairs)
        
        if input_batch_size > self.max_batch_size:
            pairs = pairs[:self.max_batch_size]
            input_batch_size = self.max_batch_size

        if self._is_xla:
            pad_batch_size = self._get_bucket_batch_size(input_batch_size)
            padded_inputs = pairs + [self.example_pair] * (pad_batch_size - input_batch_size)
        else:
            padded_inputs = pairs

        # 1. Tokenize ONCE to the absolute maximum sequence length
        # This prevents nesting errors because we only call the tokenizer once.
        inputs = self.tokenizer(
            padded_inputs,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            max_length=self.max_seq_len # Use the largest bucket size here
        )
        
        # 2. Determine the correct smaller bucket for XLA
        # Look at the attention mask to find the actual longest sequence in this batch
        actual_max_len = inputs['attention_mask'].sum(dim=1).max().item()
        target_seq_len = self._get_bucket_seq_len(actual_max_len)

        # 3. Dynamic Cropping (The "Secret Sauce")
        # If the target bucket is smaller than max_seq_len, we crop the tensors.
        # This keeps the graph static for that specific bucket size.
        if target_seq_len < self.max_seq_len:
            for key in inputs.keys():
                inputs[key] = inputs[key][:, :target_seq_len]

        # DEBUG: Check shapes before XLA execution
        # self.logger.log_info(f"Final Execution Shape: {inputs['input_ids'].shape}")

        # 4. Inference
        outputs = self._bucket_batch_inference(inputs)
        
        # 5. Process & Slice
        all_results = self._process_outputs(outputs, inputs['attention_mask'])
        return all_results[:input_batch_size]

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Config for Reranker with Query and Multiple Texts"""
        inputs = [
            {"name": "query", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "texts", "data_type": "TYPE_STRING", "dims": [-1]}
        ]
        outputs = [{"name": "scores", "data_type": "TYPE_FP32", "dims": [-1]}]

        config = auto_complete_model_config.as_dict()
        input_names = [i['name'] for i in config.get('input', [])]
        output_names = [o['name'] for o in config.get('output', [])]

        for inp in inputs:
            if inp['name'] not in input_names:
                auto_complete_model_config.add_input(inp)
        for out in outputs:
            if out['name'] not in output_names:
                auto_complete_model_config.add_output(out)

        auto_complete_model_config.set_model_transaction_policy(dict(decoupled=False))
        # Disable Triton-level batching (Dynamic Batching)
        auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config