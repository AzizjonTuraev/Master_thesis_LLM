# REaLTabFormer_generic
This is a modified version of the REaLTabFormer model for study purpose 

EXCLAMATION: 
The REaLTabFormer model belongs to Aivin V. Solatorio and Olivier Dupriez. 
Here, I am using the model for study purposes only and any credits regarding the original model should be to the paper authors.

The original paper: https://arxiv.org/abs/2302.02041

The usage of the model is the same with the original one as described here: https://github.com/worldbank/REaLTabFormer/tree/main?tab=readme-ov-file

The only difference is we can call our desired LLM model, as long as the LLM model has a model with a language modeling head on top, which can be usually called __LLM__ForCausalLM or __LLM__LMHeadModel.

In the original model, everything is set up fixedly to DistilGPT2 with 6 layers instead of 12. While calling this generic model, we can provide 3 additional parameters: 
- generic_Config, 
- generic_LMHeadModel, 
- my_config.

If we do not call these parameters, then the model uses the full GPT2 model (with 12 layers) as a default without reducing the layers. You also need to import the LLM Config and LLM Model from transformers package by yourself if you want to implement other than GPT2. If you want to use the default model, then you do not need to import the model from transformers package.

Note, while changing the LLM model architecture, take additional care with the LLM architecture itself. Because all LLM might have their unique architecture and parameters. For more details, visit https://huggingface.co/docs/transformers


```bash

## Please install these required packages first before downloading the generic realtabformer

pip install datasets==2.18.0
pip install torch==2.2.0
pip install scikit-learn==1.4.1.post1
pip install transformers==4.39.0
pip install shapely>=2.0
pip install accelerate==0.28.0

```


The default call:

```bash

pip install -i https://test.pypi.org/simple/ generic-realtabformer==1.0.3138

import pandas as pd
from generic_realtabformer.realtabformer import REaLTabFormer
from transformers.models.gptj import GPTJConfig, GPTJForCausalLM


df = pd.read_csv("foo.csv")

rtf_model = REaLTabFormer(
    generic_Config = GPTJConfig,
    generic_LMHeadModel = GPTJForCausalLM,
    my_config = {'n_layer': 12, 
                'n_positions': 1024},
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=100)

rtf_model.fit(df)
rtf_model.save("rtf_model/")
samples = rtf_model.sample(n_samples=len(df))

# Load the saved model. The directory to the experiment must be provided. 
# Additionally, provide the initial model so that our reloaded model gets the generic Config 
# and generic LMHeadModel from it directly
# You can create any new instance with the same architecture with raw parameter weights. 
# Updated parameter weights will be retrieved from the saved model, rtf_model

rtf_model2 = REaLTabFormer(
    generic_Config = GPTJConfig,
    generic_LMHeadModel = GPTJForCausalLM,
    my_config = {'n_layer': 12, 
                'n_positions': 1024},
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=100)

rtf_model2 = REaLTabFormer.load_from_dir(
    path="rtf_model/idXXXX", rtf_model2)


```