from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

# Example text for sentiment analysis
text = "I love this product, it's amazing!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

# Interpret the result
labels = ["Negative", "Positive"]
print(f"Text: {text}")
print(f"Predicted sentiment: {labels[predicted_class]} (Confidence: {probabilities[0][predicted_class]:.2f})")

# Save the model
model.save_pretrained("bert_sentiment_classifier")
tokenizer.save_pretrained("bert_sentiment_classifier")
print("Model and tokenizer saved in bert_sentiment_classifier directory")
