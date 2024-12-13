# CS410_Final_Project
# Video Link:
https://mediaspace.illinois.edu/media/t/1_cxilvqz6


## Package Installation
To install all required packages, you can use the environment file I provided as Final_Project_CS410.yaml and import it. All the packages were also specified in the code via import. I would suggest to use environment file, then run the code to see if anything got missed.

## Running Code
### Train Model
Part I: Pull source data twitter financial news sentiment from zeroshot
```python
from transformers import BertTokenizer

# Preprocessing function
def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+', '[URL]', tweet)  # Replace URLs with [URL]
    tweet = re.sub(r'#', '', tweet)  # Remove '#' from hashtags
    tweet = re.sub(r'([!?.])\1+', r'\1', tweet)  # Normalize repeated punctuation
    return tweet

# Load dataset
splits = {'train': 'sent_train.csv', 'validation': 'sent_valid.csv'}
train_df = pd.read_csv("hf://datasets/zeroshot/twitter-financial-news-sentiment/" + splits["train"])
validation_df = pd.read_csv("hf://datasets/zeroshot/twitter-financial-news-sentiment/" + splits["validation"])

# Apply preprocessing
train_df['processed_text'] = train_df['text'].apply(preprocess_tweet)
validation_df['processed_text'] = validation_df['text'].apply(preprocess_tweet)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

# Tokenize the preprocessed text
train_encodings = tokenizer(list(train_df['processed_text']), padding=True, truncation=True, max_length=128)
validation_encodings = tokenizer(list(validation_df['processed_text']), padding=True, truncation=True, max_length=128)
train_encodings['labels'] = train_df['label']
validation_encodings['labels'] = validation_df['label']
# Ready for training

```
Part II: split into train and validation dataset
```python
train_dataset = Dataset.from_dict(train_encodings)
validation_dataset = Dataset.from_dict(validation_encodings)

```
Part III: Start training and fine-tuning the model.
This will take a bit long time roughly two hours on my laptop, so I saved the trainer into my local laptop. And later on I will directly use them from my local drive. I have tried to upload all three files, but there is one >400MB, Github is preventing me from uploading it.
```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis", num_labels=3)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset
)

trainer.train()
trainer.save_model('/Users/yuchen/Downloads/') 
```
### Clean and Preprocess twitter data
This is my main challenge in this project that free open source of twitter API only contains limited amount of twitter data, so I chose to use existed twitter data from kaggle that contained NVDA tweets.
```python
df_nvda = pd.read_csv('/Users/yuchen/Downloads/Nvidia-Tweets.csv')
df_nvda = df_nvda.dropna(subset=['Text'])  # Drop rows with missing 'Text' or 'Tweet Id'

import contractions

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+', '[URL]', tweet)  # Replace URLs with [URL]
    tweet = re.sub(r'#', '', tweet)  # Remove '#' from hashtags
    tweet = re.sub(r'([!?.])\1+', r'\1', tweet)  # Normalize repeated punctuation
    tweet = tweet.lower()  # Convert to lowercase
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)  # Remove non-alphabetic characters
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet) #remove emojis
    tweet = contractions.fix(tweet)  # Expand contractions
    return tweet


df_nvda['Cleaned_Text'] = df_nvda['Text'].astype(str).apply(preprocess_tweet)


# Attempt to convert 'Datetime' column to datetime, invalid entries will become NaT (Not a Time)
df_nvda['Datetime'] = pd.to_datetime(df_nvda['Datetime'], errors='coerce')

# Drop rows where 'Datetime' could not be converted (NaT values)
df_nvda = df_nvda.dropna(subset=['Datetime'])

# Proceed with extracting the date
df_nvda['date'] = df_nvda['Datetime'].dt.date

# Check rows where Datetime is invalid (NaT)
invalid_rows = df_nvda[df_nvda['Datetime'].isna()]
print(invalid_rows)


# Get the oldest (earliest) date
oldest_date = df_nvda['date'].min()

# Get the most recent date
most_recent_date = df_nvda['date'].max()

print(f"Oldest Date: {oldest_date}")
print(f"Most Recent Date: {most_recent_date}")

df_nvda = df_nvda[~df_nvda['Datetime'].dt.weekday.isin([5, 6])]

# If you have a list of holidays, e.g., holidays in the format of datetime objects
holidays = pd.to_datetime(['2024-12-25', '2024-01-01'])  # Example of holidays
df_nvda = df_nvda[~df_nvda['date'].isin(holidays)]
```
### Use Pretrained Model to predict Sentiment
I used 3 different models in this section, first one is the BERT model that I fine-tuned. This part will take about 3 hours to complete. Here, I also used the model parameters I saved in previous part, you will modify the model_path in your case where you saved the model during testing.
```python
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Specify the path to your saved model and tokenizer files
model_path = "/Users/yuchen/Downloads/"

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path+'config.json')
model = BertForSequenceClassification.from_pretrained(model_path)

# If using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare your dataset
input_data = df_nvda['Cleaned_Text'].tolist()
# Tokenize the entire dataset
inputs = tokenizer(input_data, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Convert the tokenized data into a TensorDataset
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])

# Define DataLoader to create batches
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# List to store predictions for each batch
from tqdm import tqdm

# List to store predictions for each batch
predictions = []

# Process each batch with tqdm to track progress
for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
    input_ids, attention_mask = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Run the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Extract logits
    logits = outputs.logits
    
    # Convert logits to probabilities using softmax (for classification tasks)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get predicted class (for classification tasks)
    predicted_class = torch.argmax(probabilities, dim=-1)
    
    # Append predictions to the list
    predictions.extend(predicted_class.cpu().numpy())

# Save predictions to a DataFrame (or list of predicted classes)
df_predictions = pd.DataFrame(predictions, columns=['Predicted_Class'])

# Optionally, you can add the predictions to the original DataFrame (df_nvda)
df_nvda['Predicted_Class'] = df_predictions['Predicted_Class']

# Save to CSV or other file formats
df_nvda.to_csv('nvda_predictions.csv', index=False)
```
Next I started to load finance data from yahoo finance for NVDA stock from a start date of 2022-11-21 to end_date of 2023-02-06 because this is the start and end date of the NVDA tweets csv that I used.
```python
# Define the stock symbol and date range
symbol = 'NVDA'
start_date = '2022-11-21'
end_date = '2023-02-06'

# Get the stock data from Yahoo Finance
stock_data = yf.download(symbol, start=start_date, end=end_date)

# Select the 'Adj Close' column
adj_close = stock_data['Adj Close']

# Drop non-trading days (if any)
adj_close = adj_close.dropna()
```
In below part, I used pretrained models TextBlob and VADER sentiment analyzer to predict sentiment on the same data.
```python
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # returns a score between -1 and 1

df_nvda['Sentiment'] = df_nvda['Cleaned_Text'].apply(get_sentiment)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis to the 'Cleaned_Text' column
df_nvda['Sentiment_Score'] = df_nvda['Cleaned_Text'].apply(lambda tweet: sia.polarity_scores(tweet)['compound'])
df_nvda['Predicted'] = df_nvda['Predicted_Class'].apply(lambda score: 1 if score == 1 else (-1 if score == 0 else 0))  
daily_data = df_nvda.groupby('date').agg({
    'Predicted':'sum',
    'Sentiment': 'sum',
    'Sentiment_Score':'sum',
    'Tweet Id': 'count'  # Count the number of tweets
}).reset_index()
```
Then I group these results by date to produce daily sentiment scores by summing up all the scores in a day. And merge them with the adjusted close prices I obtained from Yahoo finance. I also created a price movement indicator.
```python
# Classify sentiment

daily_data['SentimentScore_BLOB'] = daily_data['Sentiment']#.apply(lambda score: 1 if score > 0 else (-1 if score < 0 else 0))  
daily_data['SentimentScore_VADER'] = daily_data['Sentiment_Score']#.apply(lambda score: 1 if score > 0 else (-1 if score < 0 else 0))  
daily_data['SentimentScore_BERT'] = daily_data['Predicted']
daily_data.date = pd.to_datetime(daily_data.date)
adj_close.index = pd.to_datetime(adj_close.index) 
merged_df = pd.merge(daily_data, adj_close, left_on='date', right_index=True, how='left')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
merged_df['price move'] = merged_df['Adj Close'].diff().shift(-1)
merged_df = merged_df.dropna()

X = merged_df[['Adj Close']]
y = merged_df['price move'].apply(lambda x: 1 if x > 0 else 0)
```
### Evaluate
This part is to use RandomForest to predict stock price movement and generate evaluations for each model.
```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

X = merged_df[['SentimentScore_BLOB','Adj Close']]
y = merged_df['price move'].apply(lambda x: 1 if x > 0 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

X = merged_df[['SentimentScore_BLOB','Adj Close']]
y = merged_df['price move'].apply(lambda x: 1 if x > 0 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

X = merged_df[['SentimentScore_BERT','Adj Close']]
y = merged_df['price move'].apply(lambda x: 1 if x > 0 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

merged_df['daily_return'] = merged_df['Adj Close'].pct_change()

# Assume starting capital
initial_capital = 1000
merged_df['daily_profit'] = initial_capital * merged_df['daily_return']  # Profit for fixed capital
merged_df['cumulative_profit'] = merged_df['daily_profit'].cumsum()

# Metrics
total_profit = merged_df['daily_profit'].sum()
average_daily_return = merged_df['daily_return'].mean()
win_rate = (merged_df['daily_profit'] > 0).mean() * 100
max_drawdown = (merged_df['cumulative_profit'].cummax() - merged_df['cumulative_profit']).max()

# Print results
print(f"Total Profit: ${total_profit:.2f}")
print(f"Average Daily Return: {average_daily_return:.2%}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Maximum Drawdown: ${max_drawdown:.2f}")
```
## Printed Results
No improvement can be found in all the three models in terms of F1 score and accuracy. They all had same directional predictions on prices.
             precision    recall  f1-score   support

           0       0.25      0.25      0.25         4
           1       0.50      0.50      0.50         6

    accuracy                           0.40        10
   macro avg       0.38      0.38      0.38        10
weighted avg       0.40      0.40      0.40        10

              precision    recall  f1-score   support

           0       0.00      0.00      0.00         4
           1       0.43      0.50      0.46         6

    accuracy                           0.30        10
   macro avg       0.21      0.25      0.23        10
weighted avg       0.26      0.30      0.28        10

              precision    recall  f1-score   support

           0       0.25      0.25      0.25         4
           1       0.50      0.50      0.50         6

    accuracy                           0.40        10
   macro avg       0.38      0.38      0.38        10
weighted avg       0.40      0.40      0.40        10

              precision    recall  f1-score   support

           0       0.25      0.25      0.25         4
           1       0.50      0.50      0.50         6

    accuracy                           0.40        10
   macro avg       0.38      0.38      0.38        10
weighted avg       0.40      0.40      0.40        10




### Financial Results:
BLOB and VADER have same return as below:
Total Profit: $384.48
Average Daily Return: 0.85%
Win Rate: 56.52%
Maximum Drawdown: $244.86
BERT has 0 profit mostly because most signals are neutral. Potential hypothesis is that the dataset is a bit biased.



