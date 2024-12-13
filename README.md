# CS410_Final_Project
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
This will take a bit long time, so I also uploaded the model parameters and tokenizer that I saved which will also be used in next part.
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
### 


