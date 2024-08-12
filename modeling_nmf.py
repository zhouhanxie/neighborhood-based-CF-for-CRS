from transformers import PreTrainedModel, AutoConfig


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig, PreTrainedModel
import os
from transformers.modeling_utils import PretrainedConfig

from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)

class NMFConfig(PretrainedConfig):
    def __init__(
        self, 
        n_items=None, 
        n_item_latent_factors=None, 
        mlm_config_or_model_name_or_path=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_item_latent_factors = n_item_latent_factors
        self.n_items = n_items
        self.initial_pretrained_model = None
        if isinstance(mlm_config_or_model_name_or_path, AutoConfig):
            self.mlm_config=mlm_config_or_model_name_or_path
        elif isinstance(mlm_config_or_model_name_or_path, str):
            print('received string for mlm_config, you are probably using a pretrained model')
            self.mlm_config = AutoConfig.from_pretrained(mlm_config_or_model_name_or_path)
            self.initial_pretrained_model = mlm_config_or_model_name_or_path
            
        


    def save_pretrained(self, save_directory, push_to_hub=False,**kwargs):
        self.mlm_config.save_pretrained(os.path.join(save_directory, 'base_mlm_config'))
        tmp_mlm_config = self.mlm_config 
        self.mlm_config = None
        super().save_pretrained(save_directory, **kwargs)
        self.mlm_config = tmp_mlm_config 
        
    @classmethod 
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        mlm_config = AutoConfig.from_pretrained(os.path.join(pretrained_model_name_or_path, 'base_mlm_config'))
        config_dict = cls._dict_from_json_file(os.path.join(pretrained_model_name_or_path, 'config.json'))
        config_dict['mlm_config'] = mlm_config
        config = NMFConfig(**config_dict)
        config.mlm_config = mlm_config 
        return config, {} # HF except second param to be model kwargs, we just turn this off for 

    def to_json_string(self, *args, **kwargs):
        tmp_mlm_config = self.mlm_config 
        self.mlm_config = None
        if isinstance(tmp_mlm_config, AutoConfig):
            self.mlm_config = 'transformers.AutoConfig'
        out = super().to_json_string(*args, **kwargs)
        self.mlm_config = tmp_mlm_config 
        return out


class NMFPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = NMFConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


from huggingface_hub import PyTorchModelHubMixin
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel
class NMF(NMFPreTrainedModel):
    
    # def __init__(self, n_items, n_item_latent_factors, bert_model_name='google/bert_uncased_l-4_h-256_a-4'):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if config.initial_pretrained_model is not None:
            print(f'Attempting to initialize Bert from {config.initial_pretrained_model}')
            self.model = AutoModel.from_pretrained(config.initial_pretrained_model)
            # we set it to None after using this (so next time the model loads from save_pretrained, it's the custom saved model)
            config.initial_pretrained_model = None
        else:
            self.model = AutoModel.from_config(config.mlm_config)
        self.item_embeddings = nn.Embedding(config.n_items, config.n_item_latent_factors)
        # self.classifier = nn.Linear(config.mlm_config.hidden_size + config.n_item_latent_factors, 1)
        self.projection_layer = nn.Linear(config.mlm_config.hidden_size, config.n_item_latent_factors)
        
        # self.item_embedding_projection_layer = nn.Linear(config.n_item_latent_factors, config.n_item_latent_factors)
        # self.

        # self.classifier = nn.Linear(config.n_item_latent_factors, 1)
        

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        # Get BERT sentence embeddings
        try:
            bert_outputs = self.model(
                input_ids=input_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
        except Exception as e:
            print(input_ids.shape)
            print(token_type_ids.shape)
            print(attention_mask.shape)
            print(item_ids.shape)
            print(labels.shape)
            raise e
        bert_outputs = bert_outputs.hidden_states[-1][:,0,:] # CLS embedding
        bert_outputs = self.projection_layer(bert_outputs) # Linear projection to target dimension, [bsize, h_dim]
        logits = torch.matmul(
            self.item_embeddings.weight, bert_outputs.unsqueeze(1).transpose(1, 2)
            ).squeeze(-1) # batch_size, n_items

        loss = None
        if labels is not None:
            # Ensure labels are of the correct shape and type
            labels = labels.float()  # BCEWithLogitsLoss expects float labels
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels.long())


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


from typing import List, Union
import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import DataCollatorWithPadding
from torch.utils.data.dataloader import default_collate

class NMFDataCollator:
    def __init__(self, tokenizer):
        # We use DataCollatorWithPadding for padding input_ids, token_type_ids, and attention_mask
        self.data_collator_with_padding = DataCollatorWithPadding(
            tokenizer=tokenizer, 
            return_tensors="pt"
        )

    def __call__(self, features):
        
        labels = [feature['label'] for feature in features] if 'label' in features[0] else None

        # Use the DataCollatorWithPadding for 'input_ids', 'token_type_ids', and 'attention_mask'
        batch = self.data_collator_with_padding(features)
        if labels is not None:
            batch['labels'] = default_collate(labels)

        return batch