This is a modified version of GReaT model for study purposes only.

EXCLAMATION: The GReaT model belongs to Vadim Borisov, Kathrin Sessler, Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci. Here, I am using the model for study purposes only and any credits regarding the original model should be to the paper authors.

The modified version of the model runs as expected as the original model. The only difference is that, I twisted the LLM part in a way that, one can easily slice LLM layers as their wishes and change the LLM parameters as one wishes.

The original paper, github code and the model usage can be found on this link: https://github.com/kathrinse/be_great/tree/main

In the original model, everything is set up fixedly to GPT2 with 12 layers and to DistilGPT2 with 6 layers. While calling this generic model, we can provide only one additional parameter:

- my_config.

Here is the example of how to use the modified GReaT model.
You just need to type in the parameter my_config desired LLM parameters 

Note, while changing the LLM model architecture, take additional care with the LLM architecture itself. Because all LLM might have their unique architecture and parameters. For more details, visit https://huggingface.co/docs/transformers

```Python

pip install -i https://test.pypi.org/simple/ generic-be-great==0.0.6

from generic_be_great import GReaT
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame

model = GReaT(llm='bigcode/santacoder', 
                my_config = {"n_layer" : 1},
                batch_size=32,  
                epochs=50, 
                fp16=True)
model.fit(data)

model.save("path_to_saving_folder")
synthetic_data = model.sample(n_samples=100)

# We just need to reconstruct the architecture first, 
# then we can load the paramter weights
reloaded_model = GReaT(llm='bigcode/santacoder', 
                my_config = {"n_layer" : 1},
                batch_size=32,  
                epochs=50, 
                fp16=False # True if GPU is accessible
                )

reloaded_model = GReaT.load_from_dir("path_to_saved_model_folder")

```

alternatively, we can also use LLMs from transformers library. This is usefull, especially, when the RAM memory is less but the model is huge (EleutherAI/gpt-neox-20b) or when it is not possible to load the model by its name, such as meta-llama/CodeLlama-7b-hf.

In that case we need to provide, LLM config, LLM Causal Model and corresponding Tokenizer as shown below.

```Python

from transformers import LlamaTokenizerFast
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

model_name_or_path = "hf-internal-testing/llama-tokenizer"
tokenizer = LlamaTokenizerFast.from_pretrained(model_name_or_path)

model = GReaT(llm=LlamaForCausalLM, config=LlamaConfig, batch_size=32, tokenizer=tokenizer,
                epochs=50, my_config = {"num_hidden_layers" : 1})

```

or 

```Python

from transformers import AutoTokenizer
from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM

# Specify the model's name or path
model_name_or_path = 'EleutherAI/gpt-neox-20b'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = GReaT(llm=GPTNeoXForCausalLM, config=GPTNeoXConfig, batch_size=32, tokenizer=tokenizer,
                epochs=50, my_config = {"num_hidden_layers" : 1})

```
