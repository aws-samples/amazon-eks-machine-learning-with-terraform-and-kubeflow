import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
import triton_python_backend_utils as pb_utils
from encoder_base import EncoderBaseModel, _MODEL_ARGS_FILENAME

class TritonPythonModel(EncoderBaseModel):
    """Full Sentence Embeddings implementation extending EncoderBaseModel"""

    def _init_output_config(self):
        # Extract metadata from Triton config
        out_config = pb_utils.get_output_config_by_name(self.model_config, "embeddings")
        self.embeddings_dtype = pb_utils.triton_string_to_numpy(out_config["data_type"])
        
        # Load embedding parameters from model.json
        model_dir = pb_utils.get_model_dir()
        with open(os.path.join(model_dir, _MODEL_ARGS_FILENAME)) as f:
            properties = json.load(f)
            
        self.pooling_strategy = properties.get("pooling_strategy", "mean")
        self.normalize = properties.get("normalize", True)

    def _load_model(self, model_location):
        # For embeddings, we use the base transformer model
        return AutoModel.from_pretrained(model_location)

    def _process_outputs(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        
        device = token_embeddings.device
        attention_mask = attention_mask.to(device)
        
        if self.pooling_strategy == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling_strategy == "cls":
            # Extract the first token ([CLS]) for each sequence
            embeddings = token_embeddings[:, 0]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Move to CPU and convert to list for Triton response serialization
        return embeddings.detach().cpu().numpy().astype(self.embeddings_dtype).tolist()

    def _create_response(self, request_results):
        """
        Creates a Triton response.
        request_results: List[List[float]] - a slice of the batch results for one request.
        """
        # Convert list of vectors into a [Batch, Hidden] numpy array
        output_array = np.array(request_results, dtype=self.embeddings_dtype)
        
        # If your config.pbtxt expects [Batch, 1, Hidden], you must reshape:
        # output_array = np.expand_dims(output_array, axis=1)
        
        output_tensor = pb_utils.Tensor("embeddings", output_array)
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Standard auto-complete for embedding models"""
        inputs = [{"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]}]
        outputs = [{"name": "embeddings", "data_type": "TYPE_FP32", "dims": [-1]}]

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
        # Note: max_batch_size in config.pbtxt should be > 0 for standard batching
        return auto_complete_model_config