import pandas as pd
from generic_realtabformer.realtabformer import REaLTabFormer
import datetime
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel


df = pd.read_csv("../original_train_dataset/Customer_travel_original_train_unlabeled.csv")

rtf_model = REaLTabFormer(
    generic_Config = GPT2Config,
    generic_LMHeadModel = GPT2LMHeadModel,
    my_config = {"n_layer" : 6},
    model_type="tabular",
    batch_size=2000,
    gradient_accumulation_steps=4,
    logging_steps=100)

begin_time = datetime.datetime.now()
rtf_model.fit(df)
end_time = datetime.datetime.now()

model_name = "gpt2_6_layer_customer_travel"

rtf_model.save(f"saved_models/{model_name}")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}/samples1.csv")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}/samples2.csv")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}/samples3.csv")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}/samples4.csv")

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(f"samples/train/{model_name}/samples5.csv")


samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}/samples1.csv")

samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}/samples2.csv")

samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}/samples3.csv")

samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}/samples4.csv")

samples = rtf_model.sample(n_samples=320)
samples.to_csv(f"samples/test/{model_name}/samples5.csv")

process_time = {
    "begin_time" : begin_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : end_time.strftime("%Y-%m-%d %H:%M:%S.%f")
}

process_time = pd.DataFrame([process_time])
process_time.to_csv(f"samples/process_time/{model_name}/process_time_model_name.csv")






