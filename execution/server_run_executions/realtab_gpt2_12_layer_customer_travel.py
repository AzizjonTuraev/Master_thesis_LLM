import pandas as pd
from generic_realtabformer.realtabformer import REaLTabFormer
import datetime
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
import os

df = pd.read_csv("../original_train_dataset/Customer_travel_original_train_unlabeled.csv")

rtf_model = REaLTabFormer(
    generic_Config = GPT2Config,
    generic_LMHeadModel = GPT2LMHeadModel,
    my_config = {"n_layer" : 12,
        "n_positions":1024},
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=100)

begin_time = datetime.datetime.now()
rtf_model.fit(df)
end_time = datetime.datetime.now()

base_dir = "samples"
sub_dirs = ["train", "test", "process_time"]

for sub_dir in sub_dirs:
    os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

model_name = "gpt2_12_layer_customer_travel"

rtf_model.save(f"saved_models/{model_name}")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}_samples1.csv")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}_samples2.csv")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}_samples3.csv")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}_samples4.csv")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}_samples5.csv")


samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}_samples1.csv")

samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}_samples2.csv")

samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}_samples3.csv")

samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}_samples4.csv")

samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}_samples5.csv")

process_time = {
    "begin_time" : begin_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : end_time.strftime("%Y-%m-%d %H:%M:%S.%f")
}

process_time = pd.DataFrame([process_time])
process_time.to_csv(f"samples/process_time/{model_name}_process_time_model_name.csv")


print("Job is done")





