# Synthetic Data Generation with Advanced Language Models

## 1. Introduction
The generation of synthetic tabular data has garnered significant attention due to its applications in fields such as data privacy, machine learning, and simulation. In this work, we evaluate two state-of-the-art models—Realtabformer and GReaT—for their efficacy in generating synthetic tabular data. Our study builds on these models by modifying them to utilize a broader range of large language models (LLMs), including GPT and LLaMA architectures. Additionally, we examine whether reducing the number of layers in these models can maintain high performance while lowering computational requirements.

The objective is to determine the most efficient model in generating synthetic data across multiple datasets, focusing on performance trade-offs when reducing model complexity. This work also explores the possibility of training and deploying these models locally on hardware with limited resources, such as laptops with 16GB of RAM.

## 2. Methodology
### 2.1. Model Architectures
The original implementations of Realtabformer and GReaT rely exclusively on GPT-2 with 6 or 12 layers. For this study, we re-engineered both models to be more flexible, allowing them to operate with a range of LLM architectures. The following models were used in our experiments:

GPT-2: 6 and 12 layers
GPT-NeoX: 1 layer
GPT-Neo: 2, 4, 6, and 8 layers
GPT-J: 1 layer
GPT-BigCode: 6 and 12 layers
LLaMA: 1 and 2 layers
By employing various models and layer configurations, we aim to test not only the efficacy of different LLMs but also the impact of model complexity on synthetic data generation quality.

### 2.2. Datasets
Four datasets were selected for the evaluation, representing different domains and complexity levels:

Adult: A demographic dataset commonly used in classification tasks. '[Kaggle Linke](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)'
Customer Travel: Contains information on customer travel behaviors. '[Kaggle Linke](https://www.kaggle.com/datasets/tejashvi14/tour-travels-customer-churn-prediction)'
California Housing: A dataset with features related to housing prices. '[Kaggle Linke](https://www.kaggle.com/datasets/camnugent/california-housing-prices)'
Stroke Prediction: Medical records for stroke prediction. '[Kaggle Linke](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)'

Before feeding the models with these datasets, a thorough data cleaning process was applied. This involved handling missing values and removing inconsistencies. The datasets were then split into training and test sets to prevent data leakage in the evaluation process.

### 2.3. Implementation
The re-engineered models are available for easy installation via the following commands:

```
pip install -i https://test.pypi.org/simple/ generic-realtabformer==1.0.3136
pip install -i https://test.pypi.org/simple/ generic-be-great==0.0.8
```

The documentation of the generic Realtabformer: https://test.pypi.org/project/generic-realtabformer/
The documentation of the generic GReaT: https://test.pypi.org/project/generic-be-great/
To avoid conflicts during the installation of Realtabformer, the following dependencies should be installed beforehand:

```
pip install datasets==2.18.0
pip install torch==2.2.0
pip install scikit-learn==1.4.1.post1
pip install transformers==4.39.0
pip install shapely>=2.0
```

### 2.4. Evaluation Environment
All models were trained on a high-performance computing (HPC) server provided by Oldenburg University. The server’s specifications can be found '[here](https://uol.de/fk5/wr/hochleistungsrechnen/hpc-facilities/rosa)'

## 3. Results and Discussion
### 3.1. Performance Metrics
The models were evaluated based on the following criteria:

Data Fidelity: How closely the synthetic data resembles the real dataset.
Model Scalability: Ability to run on hardware with limited resources (e.g., laptops with 16GB RAM).
Computational Efficiency: The time and resources required for model training and data generation.
Generalization: The ability to generate synthetic data across diverse datasets without overfitting.
We found that reducing the number of layers in the LLM architectures did result in faster processing times and lower resource usage, while maintaining a high level of accuracy in most cases. However, the performance varied depending on the complexity of the dataset, with more intricate datasets like Stroke benefitting from deeper models.

## 4. Conclusion
....

5. References
Aivin V. Solatorio, Olivier Dupriez: REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers. https://arxiv.org/abs/2302.02041
Vadim Borisov, Kathrin Seßler, Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci: Language Models are Realistic Tabular Data Generators. https://arxiv.org/abs/2210.06280

