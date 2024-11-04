############################################################
# 1. import key libraries
############################################################

import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

############################################################
# 2. add header and sidebar & load the data
############################################################

st.title("Tweet sentiment analysis app - US airlines")
st.markdown(" This application is a Streamlit dashboard to analyze the sentiments of tweets")

st.sidebar.title("Controls for interacting with charts")

@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv("./data/Tweets_v1.csv")
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()


############################################################
# 3. add random tweet based on sentiment selection in sidebar
############################################################

st.sidebar.subheader("1. Show random tweet")
random_tweet = st.sidebar.radio('Pick the tweet sentiment category', 
                                list(data['airline_sentiment'].unique()))

st.subheader("1. Example of a randomly selected %s sentiment tweet" %(str.upper(random_tweet)))
st.markdown(data[data['airline_sentiment']==random_tweet][["text"]].sample(n=1).iat[0,0])
                              
############################################################
# 4. show a chart of number of tweets of a particular sentiment
############################################################

st.sidebar.subheader("2. Bar chart for # of tweets")
chart_type = st.sidebar.selectbox("Pick visualization type", 
                                  ['Histogram', 'Pie Chart'], key='1')

sentiment_count = data.groupby(by='airline_sentiment', as_index=False).agg({'tweet_id':'count'})

if not st.sidebar.checkbox("Hide", True):
    st.subheader("2. Number of tweets by sentiment")
    if chart_type == "Histogram":
        fig = px.bar(sentiment_count, x='airline_sentiment', y='tweet_id', color='tweet_id', 
                     height=400, width=800, labels={"airline_sentiment": "Tweet Sentiment", 
                                                    "tweet_id": "Number of tweets"})
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='tweet_id', names='airline_sentiment',
                     height=400, width=800, labels={"airline_sentiment": "Tweet Sentiment", 
                                                    "tweet_id": "Number of tweets"})
        st.plotly_chart(fig)


############################################################
# 5. plot tweets location on the map
############################################################

st.sidebar.subheader("3. When and where are the users tweeting from?")
hour = st.sidebar.slider("Pick hour of the day", 0, 23)

rel_data = data[data['tweet_created'].dt.hour==hour]

if not st.sidebar.checkbox("Close", True):
    st.subheader("3. Tweet location by time of the day")
    st.markdown("%i tweets between %i:00 and %i:00" % (rel_data.shape[0], hour, hour+1))
    st.map(rel_data)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(rel_data)
    
        

############################################################
# 6. plot tweets count by sentiment by airline names
############################################################

st.sidebar.subheader("4. Tweets by sentiment - for selected airlines")
choice = st.sidebar.multiselect("Pick airlines",list(data['airline'].unique()), key='0')

if len(choice) > 0:
    st.subheader("4. Number of tweets by sentiment for selected airlines")
    choice_data = data[data['airline'].isin(choice)]
    fig_choice = px.histogram(choice_data, x='airline', y='airline_sentiment', histfunc='count', 
                              color='airline_sentiment', facet_col='airline_sentiment',
                              labels={'airline_sentiment':'tweet sentiment'}, height=400, width=800)
    st.plotly_chart(fig_choice)
    
    
############################################################
# 7. add wordcloud of tweets
############################################################

st.sidebar.header("5. Word Cloud")

word_sentiment = st.sidebar.radio("What sentiment?", list(data['airline_sentiment'].unique()))

if not st.sidebar.checkbox("close", True):
    st.subheader("5. Word cloud for sentiment %s" %(word_sentiment))
    words = ' '.join(data[data['airline_sentiment'] == word_sentiment]['text'])
    processed_words = ' '.join(word for word in words.split() if 'http' not in word 
                               and not word.startswith('@') and word != 'RT')
    wc = WordCloud(stopwords=STOPWORDS, background_color='white', height=500, width=800).generate(processed_words)
    plt.imshow(wc)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()
    