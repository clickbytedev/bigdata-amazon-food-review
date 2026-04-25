---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Corn Nuts are delicious, enjoyable and addictive. However, I have
    TWICE broken teeth with them, including today. Do not mindlessly chomp away at
    these snacks. I'm not being overly zealous with my suggestion that you only eat
    a few at a time and let them soften in your mouth before chewing too forcefully.
    This is the only food that has broken a tooth and it has twice!
  sentences:
  - food product quality defects completely inedible terrible taste awful disgusting
    wrong texture packaging broken
  - food triggered allergic reaction contained undisclosed allergens anaphylaxis
  - food product quality defects completely inedible terrible taste awful disgusting
    wrong texture packaging broken
- source_sentence: Or, as it says on the package Ramyun.<br /><br />Really right spicy.
    With both a (high salt and MSG) broth package and a dried vegetable packet.  The
    mushrooms in the packet never cook to my desired doneness, but they are edible.<br
    /><br />It may say how eccentric I can be that I think these are great for breakfast.<br
    /><br />Unfortunately, they were on sale when I bought them, a
  sentences:
  - food contained mold insects foreign objects glass plastic contamination infestation
  - food product quality defects completely inedible terrible taste awful disgusting
    wrong texture packaging broken
  - food triggered allergic reaction contained undisclosed allergens anaphylaxis
- source_sentence: 'Maybe I should have picked up on the name "Jerkee" but this is
    not even close to regular jerky. I don''t actually mind the label on front that
    says "nonfat dry milk added, chunked and formed" but I''m weirded out by having
    sucralose added to my jerky. I can also say that although the flavor (pepper)
    is nice, I am definitely not a fan of pink soft flat moist "jerkee."  Its texture
    is '
  sentences:
  - food contained mold insects foreign objects glass plastic contamination infestation
  - food product quality defects completely inedible terrible taste awful disgusting
    wrong texture packaging broken
  - food product quality defects completely inedible terrible taste awful disgusting
    wrong texture packaging broken
- source_sentence: I have no idea why I ordered these... must have been pretty hungry
    that day and just needed something to fill the order to get my free shipping.<br
    /><br />But speaking of shipping, I ordered these "MINI WHITE ROUNDS" and an equivalent
    order of "TATO SKINS".  The "skins" arrived unbroken while the "rounds" were about
    half rounds and half crumbles.  I think the difference was that t
  sentences:
  - food contained mold insects foreign objects glass plastic contamination infestation
  - food contained mold insects foreign objects glass plastic contamination infestation
  - food product quality defects completely inedible terrible taste awful disgusting
    wrong texture packaging broken
- source_sentence: I used My-Kap for the first time today. It works great.  I read
    where people have asked about being worried that more holes would be punched in
    the bottom of the k-cup, but you can just place the k-cup back in the holder and
    gently turn it until it pops back into the original hole from the first use.  They
    are a bit over priced in my opinion.
  sentences:
  - food product quality defects completely inedible terrible taste awful disgusting
    wrong texture packaging broken
  - food product quality defects completely inedible terrible taste awful disgusting
    wrong texture packaging broken
  - food product quality defects completely inedible terrible taste awful disgusting
    wrong texture packaging broken
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'I used My-Kap for the first time today. It works great.  I read where people have asked about being worried that more holes would be punched in the bottom of the k-cup, but you can just place the k-cup back in the holder and gently turn it until it pops back into the original hole from the first use.  They are a bit over priced in my opinion.',
    'food product quality defects completely inedible terrible taste awful disgusting wrong texture packaging broken',
    'food product quality defects completely inedible terrible taste awful disgusting wrong texture packaging broken',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.2097, 0.2097],
#         [0.2097, 1.0000, 1.0000],
#         [0.2097, 1.0000, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                         | label                                                          |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             | float                                                          |
  | details | <ul><li>min: 19 tokens</li><li>mean: 68.27 tokens</li><li>max: 145 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 15.47 tokens</li><li>max: 18 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.18</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                    | sentence_1                                                                                                                   | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>It's a good product, and yes, it is organic, but still much overpriced in my opinion. Bottles are small - only 12 ounces and taste is good but nothing special. You can buy other organic honeys out there for less.</code>                                                                                                                                                                             | <code>food product quality defects completely inedible terrible taste awful disgusting wrong texture packaging broken</code> | <code>0.0</code> |
  | <code>What's not to like? Great price, free shipping and comes to your door. My grandsons come often so we go through lots of cereal and this is one both the small kids and big kids enjoy eating.</code>                                                                                                                                                                                                    | <code>food caused severe illness food poisoning nausea vomiting diarrhea stomach ache</code>                                 | <code>0.0</code> |
  | <code>This product is steeped in water to make a tea or drink that is something like a fruit punch.  This 2-lb bag is economical and the flowers (actually a calyx, I think, if you want to be technical) are clean and fresh.  A relative of mine uses the drink for medicinal purposes and has had good results with it.  One caveat:  It stains badly if you spill it so it's best to keep it away </code> | <code>food contained mold insects foreign objects glass plastic contamination infestation</code>                             | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss",
      "cos_score_transformation": "torch.nn.modules.linear.Identity"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 24.4 seconds

### Framework Versions
- Python: 3.10.20
- Sentence Transformers: 5.4.0
- Transformers: 4.57.6
- PyTorch: 2.5.1
- Accelerate: 1.13.0
- Datasets: 2.14.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->