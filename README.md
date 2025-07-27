## Overview

The **ChildGuard** dataset is a large-scale English corpus specifically curated from X (formerly Twitter), Reddit, and YouTube to detect hate speech targeted at minors. Created entirely from scratch using platform APIs and ethical scraping methods, ChildGuard comprises 351,877 high-quality, anonymized examples. Each instance is labeled for hate presence, age group (Younger Children <11, Pre-Teens 11–12, Teens 13–17), and hate style (Explicit vs. Implicit), supporting both binary and fine-grained analyses.

## Dataset Structure

ChildGuard is organized into three subsets:

### Lexical Subset

* **Rows:** 157,280
* **Columns:** `text`, `actual_class`, `predicted_class`, `Hate` (1 = hate, 0 = non-hate)
* **Description:** Focuses on word-level features (vocabulary richness, sentiment). Omits age annotations for streamlined lexical analysis.

### Contextual Subset

* **Rows:** 194,597
* **Columns:** `text`, `actual_class`, `predicted_class`, `Age_Group`, `Hate` (1 = hate, 0 = non-hate)
* **Description:** Enriched with age group labels and discourse-level context for nuanced semantic evaluation.

### Full ChildGuard Corpus

* **Rows:** 351,877
* **Columns:** `text`, `actual_class`, `predicted_class`, `Age_Group`, `Hate` (1 = hate, 0 = non-hate)
* **Description:** Integrates both lexical and contextual annotations, covering all three age groups for comprehensive model development.

## Key Statistics

| Metric                       | Lexical | Contextual | Full Corpus |
| ---------------------------- | :-----: | :--------: | :---------: |
| Unique Words                 |  28,764 |   22,890   |    35,412   |
| Average Text Length (tokens) |   19.1  |    22.7    |     21.2    |
| Explicit Hate Proportion     |   26%   |     24%    |     25%     |
| Implicit Hate Proportion     |   12%   |     14%    |     13%     |

## Data Fields

* `text`: Raw user-generated statement.
* `actual_class`: Ground truth label (`Hate` vs. `Non-Hate`).
* `predicted_class`: Model-predicted label for evaluation.
* `Hate`: Binary indicator (1 = Hate; 0 = Non-Hate).
* `Age_Group`: Age category for contextual analysis (Teens, Pre-Teens, Younger Children).

## Usage

Download the subset(s) you need from the repository:

1. **`lexical_childhate.csv`** (157,280 rows)
2. **`contextual_childhate.csv`** (194,597 rows)
3. **`childguard_dataset.csv`** (351,877 rows)

Each file includes the columns described above. Use the lexical subset for vocabulary or sentiment studies, the contextual subset for age-aware modeling, and the full corpus for comprehensive experiments.

## Baseline Models and Source URLs

We benchmarked a variety of classical, neural, and large language models. Below, each model is listed alongside the URL from which its implementation was sourced in the Baselines section of the paper:

* **Support Vector Machine (SVM)**: [https://scikit-learn.org/1.5/modules/svm.html](https://scikit-learn.org/1.5/modules/svm.html)
* **Long Short-Term Memory (LSTM)**: [https://www.tensorflow.org/api\_docs/python/tf/keras/layers/LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
* **Convolutional Neural Network (CNN)**: [https://www.tensorflow.org/tutorials/images/cnn](https://www.tensorflow.org/tutorials/images/cnn)
* **BERT-base**: [https://huggingface.co/google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
* **RoBERTa**: [https://huggingface.co/docs/transformers/en/model\_doc/roberta](https://huggingface.co/docs/transformers/en/model_doc/roberta)
* **GPT-3.5-turbo**: [https://huggingface.co/Xenova/gpt-3.5-turbo](https://huggingface.co/Xenova/gpt-3.5-turbo)
* **GPT-4o**: [https://platform.openai.com/docs/models/gpt-4o](https://platform.openai.com/docs/models/gpt-4o)
* **GPT-4.5**: [https://openai.com/index/introducing-gpt-4-5/](https://openai.com/index/introducing-gpt-4-5/)
* **DeepSeek-V3**: [https://huggingface.co/deepseek-ai/DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)

## Results and Discussion

We evaluated nine models—SVM, LSTM, CNN, BERT, RoBERTa, GPT-3.5-turbo, GPT-4o, GPT-4.5, and DeepSeek-V3—on the Contextual, Lexical, and Full ChildGuard datasets using accuracy, precision, recall, and macro-averaged F1.

**Table 1: Dataset-level binary classification results**

| Dataset    | Model         | Accuracy | Precision | Recall |  F1  |
| ---------- | ------------- | :------: | :-------: | :----: | :--: |
| Contextual | GPT-4.5       |   61.8%  |   61.4%   |  60.9% | 61.1 |
|            | GPT-4o        |   60.3%  |   59.8%   |  59.2% | 59.5 |
|            | DeepSeek-V3   |   59.2%  |   58.7%   |  58.0% | 58.3 |
|            | GPT-3.5-turbo |   58.4%  |   58.0%   |  57.4% | 57.7 |
|            | BERT          |   56.3%  |   55.9%   |  55.1% | 55.5 |
|            | RoBERTa       |   55.1%  |   54.6%   |  54.0% | 54.3 |
|            | LSTM          |   52.6%  |   52.1%   |  51.5% | 51.8 |
|            | SVM           |   50.2%  |   49.8%   |  49.1% | 49.4 |
|            | CNN           |   49.3%  |   48.8%   |  48.2% | 48.5 |
| Lexical    | GPT-4.5       |   62.5%  |   62.0%   |  61.6% | 61.8 |
|            | GPT-4o        |   61.0%  |   60.5%   |  60.0% | 60.2 |
|            | DeepSeek-V3   |   60.1%  |   59.6%   |  59.0% | 59.3 |
|            | GPT-3.5-turbo |   59.4%  |   58.9%   |  58.2% | 58.5 |
|            | BERT          |   57.1%  |   56.6%   |  56.0% | 56.3 |
|            | RoBERTa       |   56.0%  |   55.3%   |  54.9% | 55.1 |
|            | LSTM          |   53.8%  |   53.2%   |  52.5% | 52.8 |
|            | SVM           |   51.3%  |   50.9%   |  50.2% | 50.5 |
|            | CNN           |   50.1%  |   49.6%   |  49.0% | 49.3 |
| Full       | GPT-4.5       |   60.2%  |   59.7%   |  59.0% | 59.3 |
| ChildGuard | GPT-4o        |   59.4%  |   58.9%   |  58.2% | 58.5 |
|            | DeepSeek-V3   |   58.7%  |   58.1%   |  57.4% | 57.7 |
|            | GPT-3.5-turbo |   57.5%  |   57.0%   |  56.3% | 56.6 |
|            | BERT          |   56.4%  |   56.0%   |  55.2% | 55.5 |
|            | RoBERTa       |   55.7%  |   55.0%   |  54.4% | 54.7 |
|            | LSTM          |   53.0%  |   52.5%   |  51.9% | 52.1 |
|            | SVM           |   52.0%  |   51.6%   |  51.3% | 51.5 |
|            | CNN           |   51.0%  |   50.5%   |  50.0% | 50.2 |



**Table 2: Age-group performance on the Full ChildGuard dataset**

| Age Group         | Model         | Accuracy | Precision | Recall |  F1  |
| ----------------- | ------------- | :------: | :-------: | :----: | :--: |
| Teens (13–17)     | GPT-4.5       |   60.2%  |   59.7%   |  59.0% | 59.3 |
|                   | GPT-4o        |   59.4%  |   58.9%   |  58.2% | 58.5 |
|                   | DeepSeek-V3   |   58.7%  |   58.1%   |  57.4% | 57.7 |
|                   | GPT-3.5-turbo |   58.1%  |   57.4%   |  56.8% | 57.1 |
|                   | BERT          |   56.8%  |   56.1%   |  55.5% | 55.8 |
|                   | RoBERTa       |   55.9%  |   55.2%   |  54.7% | 54.9 |
|                   | LSTM          |   53.6%  |   52.9%   |  52.3% | 52.6 |
|                   | SVM           |   52.8%  |   52.2%   |  51.7% | 51.9 |
|                   | CNN           |   52.1%  |   51.6%   |  51.0% | 51.3 |
| Pre-Teens (11–12) | GPT-4.5       |   61.1%  |   60.5%   |  59.8% | 60.1 |
|                   | GPT-4o        |   60.3%  |   59.7%   |  59.1% | 59.4 |
|                   | DeepSeek-V3   |   59.5%  |   58.8%   |  58.2% | 58.5 |
|                   | GPT-3.5-turbo |   58.9%  |   58.2%   |  57.5% | 57.8 |
|                   | BERT          |   57.4%  |   56.8%   |  56.1% | 56.4 |
|                   | RoBERTa       |   56.7%  |   56.0%   |  55.4% | 55.7 |
|                   | LSTM          |   54.8%  |   54.1%   |  53.6% | 53.8 |
|                   | SVM           |   53.6%  |   53.0%   |  52.4% | 52.7 |
|                   | CNN           |   53.0%  |   52.4%   |  51.8% | 52.1 |
| Younger (< 11)    | GPT-4.5       |   62.4%  |   61.8%   |  61.0% | 61.4 |
|                   | GPT-4o        |   61.0%  |   60.3%   |  59.6% | 59.9 |
|                   | DeepSeek-V3   |   60.2%  |   59.5%   |  58.7% | 59.1 |
|                   | GPT-3.5-turbo |   59.5%  |   58.8%   |  58.0% | 58.4 |
|                   | BERT          |   58.0%  |   57.2%   |  56.6% | 56.9 |
|                   | RoBERTa       |   57.1%  |   56.4%   |  55.7% | 56.0 |
|                   | LSTM          |   55.9%  |   55.1%   |  54.4% | 54.7 |
|                   | SVM           |   54.6%  |   54.0%   |  53.2% | 53.6 |
|                   | CNN           |   54.0%  |   53.3%   |  52.7% | 53.0 |

