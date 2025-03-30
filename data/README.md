# Data Directory

Place your datasets in this directory for training and evaluation.

## Expected File Format

The system expects CSV files with the following format:

```
utterance,category
"I need to reset my account password","ACCOUNT"
"Why am I being charged a cancellation fee?","CANCELLATION_FEE"
"How can I contact customer support?","CONTACT"
...
```

## Required Columns

- `utterance`: The customer query text (string)
- `category`: The intent category (string)

## Supported Intent Categories

The system is designed to recognize the following 11 intent categories:

1. ACCOUNT
2. CANCELLATION_FEE
3. CONTACT
4. DELIVERY
5. FEEDBACK
6. INVOICE
7. NEWSLETTER
8. ORDER
9. PAYMENT
10. REFUND
11. SHIPPING_ADDRESS

## Obtaining Datasets

You can download the customer service intent dataset:
[Customer Support Intent Dataset](https://www.kaggle.com/datasets/scodepy/customer-support-intent-dataset)

Or create your own dataset following the format above.

## Data Preprocessing

The system will automatically handle:
- Tokenization
- Padding/truncation to a fixed length (25 tokens)
- Label encoding
