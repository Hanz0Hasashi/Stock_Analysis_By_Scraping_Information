#Step 1 : Creating a API from telegram (telethon) for scraping messages from the app
#step 2 : SCraping the required messeges from  the suited or specified channel from telgram
#step 3 : DAta pre-prossesing 
#step 4 : building and training a model








#connecting to telegram through api
from telethon.sync import TelegramClient
import pandas as pd

# Your credentials
api_id = 'api_id'
api_hash = 'api_hash'
phone = 'phone_number'

# Creating a Telegram client
client = TelegramClient('session_name', api_id, api_hash)




# Function to scrape the messages from channel
async def scrape_telegram(channel_link, limit=100):
    # Connect to the client
    await client.start()
    
    # Get messages from a specific channel
    messages = []
    async for message in client.iter_messages(channel_link, limit=limit):
        if message.message:
            messages.append([message.date, message.sender_id, message.message])
    
    # Save the messages to a DataFrame
    df = pd.DataFrame(messages, columns=["Date", "Sender ID", "Message"])
    return df

channel_link = 'Place_telegram_channel_link_here'
df = client.loop.run_until_complete(scrape_telegram(channel_link, limit=200))
print(df.head())



#DAta pre-processing to clean the unwanted wods from the dataFrame
import re

def clean_text(text):
    # Removing URLs, special characters, etc.
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text

df['Cleaned_Message'] = df['Message'].apply(clean_text)



#cleaning and creating sentimental analysis by creating sentiment column Which is Important for predictions

from textblob import TextBlob

# Sentiment analysis function
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Add sentiment column
df['Sentiment'] = df['Cleaned_Message'].apply(get_sentiment)
print(df[['Cleaned_Message', 'Sentiment']].head())






#Creating model using RAndom Forest Classifier to predict stocks

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Vectorizing the text data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Cleaned_Message']).toarray()

# For Example target data (stock price movement: 0 = no movement, 1 = upward movement, -1 = downward movement)
# In a real scenario, this could be historical stock movement data.
df['Stock_Movement'] = df['Sentiment'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Stock_Movement'], test_size=0.2, random_state=42)

# Model training using RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')