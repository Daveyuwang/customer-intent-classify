# Customer Intent Recognition

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/limbo23/Customer_Intent_Recognition)

Try the live demo: [Customer Intent Recognition](https://huggingface.co/spaces/limbo23/Customer_Intent_Recognition)

A machine learning system that identifies customer intent from text queries using multiple deep learning models and LLM integration.

![Demo Screenshot](https://imgur.com/a/sKcmAF6)

## Overview

This project implements a comprehensive intent classification system using different neural network architectures:

- **BERT** (fine-tuned pre-trained Transformer)
- **TextCNN** (Convolutional Neural Network for text)
- **LSTM** (Long Short-Term Memory network)
- **Transformer** (custom implementation)
- **Combined model** (BERT+CNN+LSTM with attention)

Additionally, the system offers LLM-based intent classification through the DeepSeek API for enhanced accuracy.

## Recognized Intents

The system can identify 11 different customer intent categories:

1. **ACCOUNT** - Account-related queries
2. **CANCELLATION_FEE** - Fee cancellation questions
3. **CONTACT** - Customer support contact requests
4. **DELIVERY** - Delivery status and information
5. **FEEDBACK** - Product or service feedback
6. **INVOICE** - Invoice or billing inquiries
7. **NEWSLETTER** - Newsletter subscription management
8. **ORDER** - Order status and information
9. **PAYMENT** - Payment method or issues
10. **REFUND** - Refund requests or information
11. **SHIPPING_ADDRESS** - Shipping address updates

## Training New Models

To train the models on your own dataset:

```bash
python train.py --train_file data/your_training_data.csv --test_file data/your_test_data.csv --plot
```

Command-line options:
- `--train_file`: Path to training CSV (required)
- `--test_file`: Path to testing CSV (optional)
- `--epochs`: Number of training epochs (default: 5)
- `--output_prefix`: Prefix for saved model files
- `--plot`: Generate visualization plots during training


## Models

- **BERT**: A fine-tuned [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) model, leveraging pre-trained contextual embeddings for classification.  
- **TextCNN**: Convolutional neural network with multiple kernel sizes (3, 4, 5) to capture different n-gram features.  
- **LSTM**: Long Short-Term Memory network for sequential dependencies, with an embedding layer and a fully connected output.  
- **Transformer**: A lightweight custom encoder implementation with multi-head self-attention and feed-forward networks.

### Combined Model (BERT+CNN+LSTM)
A hybrid architecture that:
1. Processes input through BERT, CNN, and LSTM in parallel
2. Uses a custom attention mechanism to combine the features from each model
3. Applies final classification through fully connected layers

### LLM Integration

Optionally, the system can use the DeepSeek API for intent classification. A simple prompt instructs the LLM to output both a predicted class and a confidence score, complementing the local neural network models.

## Dataset

The system was trained on a customer support intent dataset containing:
- Customer queries/utterances
- Intent categories

For your own training, prepare a CSV file with the following columns:
- `utterance`: The customer query text
- `category`: The intent category (one of the 11 supported intents)


## Acknowledgments

- The models are inspired by research papers:
  - BERT: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
  - TextCNN: [Kim, 2014](https://arxiv.org/abs/1408.5882)
  - Transformers: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- DeepSeek for the LLM API access
- Hugging Face for the hosting and model libraries