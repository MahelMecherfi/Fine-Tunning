# Fine-Tunning
# GPT-2 Adaptation for Homer Simpson Dialogue Generation

## Overview

This project explores two different approaches for adapting a pre-trained language model to a specific conversational style:

- **Full Fine-Tuning**
- **LoRA (Low-Rank Adaptation)**

The goal is to adapt **GPT-2**, a pre-trained causal language model from Hugging Face, to generate English responses based on a dataset of Homer Simpson dialogues.

The project also compares the computational cost and performance of full fine-tuning and LoRA.

---

## Objective

GPT-2 is already able to generate English text because it has been pre-trained on a large text corpus.

Instead of training a language model from scratch, this project starts from the pre-trained GPT-2 model and adapts it to a new domain: conversational data inspired by Homer Simpson.

The main research question is:

> Can LoRA achieve performance close to full fine-tuning while training significantly fewer parameters?

---

# Dataset

The dataset used in this project is:

```text
Jim-Webfox/HomerSimpson
```

available through Hugging Face Datasets.

The dataset contains conversations structured approximately as:

```text
### Human: Why are you eating another donut?
### Assistant: Because donuts are delicious.
```

The objective is to teach GPT-2 to generate the assistant response given a human prompt.

The dataset is divided into three subsets:

```text
80% Training
10% Validation
10% Test
```

A fixed random seed is used to ensure that both experiments use exactly the same data split.

```python
SEED = 42
```

This is important because the comparison between Full Fine-Tuning and LoRA must be performed under the same experimental conditions.

---

# Base Model

The base model is:

```text
GPT-2
```

from Hugging Face Transformers.

GPT-2 is a **decoder-only Transformer** trained using causal language modeling.

Its main objective is to predict the next token in a sequence.

For example:

```text
Input:

Homer loves

Target:

donuts
```

More generally:

```text
Token 1 → predict Token 2
Token 2 → predict Token 3
Token 3 → predict Token 4
...
```

The same original pre-trained GPT-2 model is used as the starting point for both experiments.

---

# Tokenization

The official GPT-2 tokenizer is used.

The tokenizer transforms text into tokens and then into numerical token IDs.

Example:

```text
"Homer loves donuts"

        ↓

tokens

        ↓

token IDs

        ↓

GPT-2
```

A new tokenizer is **not trained**, because the pre-trained GPT-2 embeddings correspond to the vocabulary learned with its original tokenizer.

GPT-2 does not have a dedicated padding token by default, so the end-of-sequence token is also used as the padding token.

```python
tokenizer.pad_token = tokenizer.eos_token
```

---

# Transformer Architecture

GPT-2 is based on the Transformer architecture.

Each Transformer block contains two main components:

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
Feed Forward Network
  ↓
Residual Connection
```

The attention mechanism relies on three representations:

```text
Q = Query
K = Key
V = Value
```

They are obtained using learned projections:

```text
Q = XWq
K = XWk
V = XWv
```

The attention operation is:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dk)V
```

Because GPT-2 is an autoregressive language model, it uses a **causal mask**.

A token can attend to previous tokens but cannot see future tokens.

---

# Experiment 1 — Full Fine-Tuning

In the first experiment, the complete GPT-2 model is fine-tuned on the Homer Simpson dialogue dataset.

All model parameters remain trainable.

Conceptually:

```text
Pre-trained GPT-2
        ↓
Homer Simpson Dataset
        ↓
Update all GPT-2 parameters
        ↓
Fine-tuned GPT-2
```

All Transformer layers can therefore adapt to the new dataset.

Example:

```python
for parameter in model.parameters():
    parameter.requires_grad = True
```

### Advantages

Full fine-tuning provides maximum flexibility because all model parameters can change.

### Disadvantages

It requires:

- more GPU memory;
- more computation;
- more trainable parameters;
- more storage for the final model.

The full modified GPT-2 model must also be saved after training.

---

# Experiment 2 — LoRA

The second experiment uses **LoRA — Low-Rank Adaptation**.

Instead of modifying all GPT-2 parameters, the original model parameters are mostly frozen.

Small trainable matrices are added to selected layers of the Transformer.

For a model weight matrix:

```text
W
```

LoRA introduces an update:

```text
ΔW = BA
```

Therefore:

```text
W' = W + BA
```

where `A` and `B` are much smaller matrices.

For example, instead of learning a large matrix of size:

```text
768 × 768
```

LoRA can use a low rank such as:

```text
r = 8
```

with two matrices approximately shaped as:

```text
768 × 8

and

8 × 768
```

This drastically reduces the number of trainable parameters.

---

## LoRA Parameters

Typical LoRA parameters used in the experiment include:

```text
rank (r)
lora_alpha
lora_dropout
```

### Rank

The rank controls the capacity of the LoRA adaptation.

A small rank means fewer trainable parameters.

For example:

```text
r = 8
```

### LoRA Alpha

`lora_alpha` controls the scaling of the LoRA update.

Conceptually:

```text
W' = W + (alpha / r) BA
```

### LoRA Dropout

Dropout can be applied to the LoRA branch during training to reduce overfitting.

---

# Full Fine-Tuning vs LoRA

The key difference between the two experiments is the number of parameters that are updated.

```text
                  GPT-2 Pre-trained
                         |
              -----------------------
              |                     |
              |                     |
      Full Fine-Tuning            LoRA
              |                     |
      All parameters         GPT-2 mostly frozen
         updated                  +
                           small LoRA matrices
                                updated
```

Both methods are therefore forms of **fine-tuning**.

LoRA is more specifically a form of:

```text
Parameter-Efficient Fine-Tuning
```

or **PEFT**.

---

# Training

The model is trained using **causal language modeling**.

For every sequence, the model predicts the next token.

The loss used is cross-entropy.

Conceptually:

```text
Model prediction
       ↓
Probability distribution over vocabulary
       ↓
Compare with correct next token
       ↓
Cross-Entropy Loss
       ↓
Backpropagation
       ↓
Parameter update
```

The optimizer updates the trainable parameters according to the gradients calculated during backpropagation.

---

# Evaluation

Both models are evaluated using the same validation and test datasets.

The main quantitative metrics are:

### Loss

Cross-entropy loss measures how well the model predicts the correct next tokens.

Lower is better.

### Perplexity

Perplexity is calculated from the language modeling loss:

```text
Perplexity = exp(Loss)
```

Lower perplexity generally indicates better next-token prediction.

---

# Computational Metrics

The benchmark also compares the computational efficiency of both approaches.

The following values can be measured:

```text
Total parameters
Trainable parameters
Percentage of trainable parameters
Training time
Peak GPU memory usage
Validation loss
Test loss
Test perplexity
Saved model size
```

This allows the project to compare not only model quality but also computational cost.

---

# Qualitative Evaluation

The models can also be evaluated using identical prompts.

For example:

```text
Human: Homer, what do you think about donuts?
Assistant:
```

or:

```text
Human: Homer, why don't you want to go to work?
Assistant:
```

The same generation parameters should be used for both models.

Example:

```text
max_new_tokens = 50
temperature = 0.8
top_p = 0.9
```

This makes the qualitative comparison more meaningful.

---

# Expected Benchmark

The final results can be summarized in a table such as:

| Metric | Full Fine-Tuning | LoRA |
|---|---:|---:|
| Total Parameters | — | — |
| Trainable Parameters | — | — |
| Trainable Parameters (%) | — | — |
| Training Time | — | — |
| Peak GPU Memory | — | — |
| Validation Loss | — | — |
| Test Loss | — | — |
| Test Perplexity | — | — |
| Saved Model Size | — | — |

The values are filled after running the experiments.

---

# Why This Benchmark Is Interesting

Full fine-tuning potentially provides greater adaptation capacity because every model parameter can change.

However, it can be expensive.

LoRA attempts to achieve similar adaptation using only a small number of additional trainable parameters.

The benchmark therefore studies the trade-off between:

```text
Model quality
        vs
Training cost
        vs
Number of trainable parameters
        vs
GPU memory
        vs
Storage
```

A particularly interesting result would be if LoRA achieves validation/test performance close to full fine-tuning while requiring significantly fewer trainable parameters.

---

# Technologies

The project uses:

```text
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
PEFT / LoRA
```

Main libraries:

```bash
pip install torch transformers datasets peft accelerate
```

---

# Basic Workflow

The complete workflow is:

```text
1. Load Homer Simpson dataset

2. Clean and format dialogues

3. Split dataset into:
   - training
   - validation
   - test

4. Load GPT-2 tokenizer

5. Tokenize the dataset

6. Load original pre-trained GPT-2

7. Full Fine-Tuning
   ↓
   Evaluate
   ↓
   Save model

8. Reload original pre-trained GPT-2

9. Add LoRA adapters
   ↓
   Train LoRA parameters
   ↓
   Evaluate
   ↓
   Save adapters

10. Compare both experiments
```

An important detail is that the LoRA experiment must **not** start from the fully fine-tuned model.

Both experiments must independently start from the same original GPT-2 checkpoint:

```text
                    Original GPT-2
                          |
                 -------------------
                 |                 |
                 ↓                 ↓
         Full Fine-Tuning        LoRA
```

---

# Conclusion

This project demonstrates two approaches for adapting a pre-trained Transformer language model to a specialized conversational dataset.

Full fine-tuning updates the complete model, while LoRA keeps most of the original GPT-2 parameters frozen and learns a small number of additional parameters.

The final benchmark evaluates whether LoRA can provide a good trade-off between adaptation quality and computational efficiency when adapting GPT-2 to the Homer Simpson dialogue dataset.
