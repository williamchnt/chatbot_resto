
import random
import pandas as pd
import joblib
# Importing modules
# warnings.filterwarnings("ignore")
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import re
from difflib import SequenceMatcher

# linear kernel
from sklearn.metrics.pairwise import linear_kernel

import streamlit as st
from streamlit_chat import message as st_message

def setIconPage():
    st.set_page_config(
        page_title = "Chat bot - j'recommande ton resto",
        layout = 'wide'
    )

def HideStreamlitContent():
    
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)



@st.cache_data
def read_data():
    df = pd.read_csv('Data_Restaurants.csv', sep = '|')
    df = df.drop(['Numero de telephone'], axis=1)
    df.drop_duplicates(inplace=True)

    df['Specialite'] = df['Specialite'].str.lower()
    df['Specialite'] = df['Specialite'].str.replace(' ', '_')
    df['Region'] = df['Region'].str.lower()
    df['Region'] = df['Region'].str.replace(' ', '_')
    df['Nom du restaurant'] = df['Nom du restaurant'].str.lower()
    df['Nom du restaurant'] = df['Nom du restaurant'].str.replace(' ', '_')


    stop = stopwords.words('french')
    # remove punctuation
    df['Specialite'] = df['Specialite'].str.replace('[^\w\s]','')
    df['Region'] = df['Region'].str.replace('[^\w\s]','')
    df['Nom du restaurant'] = df['Nom du restaurant'].str.replace('[^\w\s]','')
    df['Specialite'] = df['Specialite'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['Region'] = df['Region'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['Nom du restaurant']  = df['Nom du restaurant'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return df 

def recommend_restaurant(name):
    name = name.lower()
    name = name.replace(' ', '_')
    name = name.replace('[^\w\s]','')
    words = name.split()
    # Remove stop words
    filtered_words = [word for word in words if word not in stopwords.words('french')]
    # Join the filtered words back into a sentence
    name = ' '.join(filtered_words)
    indices = pd.Series(df_percent.index)
    # cosine_similarities = joblib.load('recommend_name_restaurant.pkl')
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    try:
        idx = indices[indices == name].index[0]
    except:
        print('Restaurant non trouvé')
        return None
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
# Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['Specialite', 'Note moyenne', 'Indicateur de prix','Ville','Departement','Region'])
    
    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:

        df_new = df_new.append(pd.DataFrame(df_percent[['Specialite','Note moyenne', 'Indicateur de prix','Ville','Departement','Region']][df_percent.index == each].sample()))
        
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['Specialite','Note moyenne', 'Indicateur de prix','Ville','Departement','Region'], keep=False)
    df_new = df_new.sort_values(by='Note moyenne', ascending=False)

    
    print('TOP %s RESTAURANTS SIMILAIRE A  %s AVEC UNE NOTE PROCHE: ' % (str(len(df_new)), name))
    
    return df_new

def recommend_region(name, region=None):
    name = name.lower()
    name = name.replace(' ', '_')
    name = name.replace('[^\w\s]','')
    words = name.split()
    # Remove stop words
    filtered_words = [word for word in words if word not in stopwords.words('french')]
    # Join the filtered words back into a sentence
    name = ' '.join(filtered_words)
    indices = pd.Series(df_percent.index)
    # cosine_similarities = joblib.load('recommend_name_restaurant.pkl')
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    try:
        # Find the index of the restaurant entered
        idx = indices[indices == name].index[0]
    except:
        print('Restaurant non trouvé')
        return None
    
    # Find the restaurants with a similar cosine-sim value and order them from biggest number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Filter dataframe by region if specified
    if region:
        df_new = df_percent[df_percent['Region'] == region].copy()
    else:
        df_new = df_percent.copy()
    
    # Create the top 30 similar restaurants with some of their columns
    df_new = df_new[df_new.index.isin(recommend_restaurant)][['Specialite', 'Note moyenne', 'Indicateur de prix','Ville','Departement','Region']]
        
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['Specialite','Note moyenne', 'Indicateur de prix','Ville','Departement','Region'], keep=False)
    df_new = df_new.sort_values(by='Note moyenne', ascending=False)

    print('TOP %s RESTAURANTS SIMILAIRES A %s AVEC UNE NOTE PROCHE : ' % (str(len(df_new)), name))
    
    return df_new

def recommend_departement(name, departement=None):
    name = name.lower()
    name = name.replace(' ', '_')
    name = name.replace('[^\w\s]','')
    words = name.split()
    # Remove stop words
    filtered_words = [word for word in words if word not in stopwords.words('french')]
    # Join the filtered words back into a sentence
    name = ' '.join(filtered_words)
    indices = pd.Series(df_percent.index)
    # cosine_similarities = joblib.load('recommend_name_restaurant.pkl')
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    try:
        # Find the index of the restaurant entered
        idx = indices[indices == name].index[0]
    except:
        print('Restaurant non trouvé')
        return None
    
    # Find the restaurants with a similar cosine-sim value and order them from biggest number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Filter dataframe by department if specified
    if departement is not None:
        df_new = df_percent[df_percent['Departement'] == departement].copy()
    else:
        df_new = df_percent.copy()
    
    # Create the top 30 similar restaurants with some of their columns
    df_new = df_new[df_new.index.isin(recommend_restaurant)][['Specialite', 'Note moyenne', 'Indicateur de prix','Ville','Departement','Region']]
        
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['Specialite','Note moyenne', 'Indicateur de prix','Ville','Departement','Region'], keep=False)
    df_new = df_new.sort_values(by='Note moyenne', ascending=False)

    print('TOP %s RESTAURANTS SIMILAIRES A %s AVEC UNE NOTE PROCHE DANS LE DEPARTEMENT %s: ' % (str(len(df_new)), name, str(departement)))
    
    return df_new

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

#function to find between each word in the name of the restaurant and the name of the restaurant in the database
def find_resto(word):
    for i in df_percent.index:
        # check similarity between the word and the name of the restaurant
        similarity = similar(i, word)
        if similarity > 0.9:
            return i
            
def find_region(word):
    for i in df_percent['Region'].unique():
        # check similarity between the word and the name of the region
        similarity = similar(i, word)
        if similarity > 0.9:
            return i
        
def find_departement(word):
    pattern = r'\d+'
    if re.findall(pattern, word):
        for i in df_percent['Departement'].unique():
            # check similarity between the word and the name of the region
            if word == str(i):
                return i
            
def is_resto(word):
    for i in df_percent.index:
        # check similarity between the word and the name of the restaurant
        similarity = similar(i, word)
        if similarity > 0.9:
            return True
            
    return False
            
def is_region(word):
    for i in df_percent['Region'].unique():
        # check similarity between the word and the name of the region
        similarity = similar(i, word)
        if similarity > 0.9:
            return True

    return False

def is_departement(word):
    pattern = r'\d+'
    if re.findall(pattern, word):
        for i in df_percent['Departement'].unique():
            # check similarity between the word and the name of the region
            if word == str(i):
                return True
            
    return False


def print_resto(df,max=1000000):
    if len(df) == 0:
        return "Je n'ai pas trouvé de restaurant correspondant à votre demande"
    else:
        finalAnswer = str("Voici les restaurants que je vous recommande : \n")
        if len(df) < max:
            for i in range(len(df)):
                resto_string = str(i)+" : " +str('Spécialité : '+df.iloc[i]['Specialite']) + ', Note (/5) : ' + str(df.iloc[i]['Note moyenne']) + ', Prix : ' + str(df.iloc[i]['Indicateur de prix']) + ', Ville : ' + str(df.iloc[i]['Ville']) + ', Departement : ' + str(df.iloc[i]['Departement']) + ', Région : ' + str(df.iloc[i]['Region']) 
                finalAnswer += resto_string + "\n"
        else:
            for i in range(max):
                resto_string = str(i)+" : " +str('Spécialité : '+df.iloc[i]['Specialite']) + ', Note (/5) : ' + str(df.iloc[i]['Note moyenne']) + ', Prix : ' + str(df.iloc[i]['Indicateur de prix']) + ', Ville : ' + str(df.iloc[i]['Ville']) + ', Departement : ' + str(df.iloc[i]['Departement']) + ', Région : ' + str(df.iloc[i]['Region']) 
                finalAnswer += resto_string + "\n"

    return finalAnswer

def word_is_max(word):
    if  "max" in word:
        return True
    else:
        return False


def generate_response(user_input):
    restoFind = None
    regionFind = None
    departementFind = None
    max = 1000000
    for word in user_input.split():
            if word_is_max(word):
                max = int(word[4:])
            if is_resto(word):
                restoFind = find_resto(word)
            if is_region(word):
                regionFind = find_region(word)
            if is_departement(word):
                departementFind = find_departement(word)

    if restoFind != None:
        if restoFind != None:
            return print_resto(recommend_region(restoFind, regionFind),max)
        elif departementFind != None:
            return print_resto(recommend_departement(restoFind, departementFind),max)
        else:
            return print_resto(recommend_restaurant(restoFind),max)
    else:
        return "Je n'ai pas compris votre demande, veuillez réessayer"

# MAIN

setIconPage()
HideStreamlitContent()

df = read_data()
stop = stopwords.words('french')
df_percent = df.sample(frac=0.2)
df_percent.set_index('Nom du restaurant', inplace=True)
indices = pd.Series(df_percent.index)


# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['Specialite'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

greeting_inputs = ['bonjour', 'salut', 'hello', 'hey', 'coucou', 'yo', 'bonsoir', 'bonjour', 'salut', 'hello', 'hey', 'coucou', 'yo', 'bonsoir']
# greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup",'hello')
greeting_responses = ["hey", "Comment puis-je vous aider ?", "*nods*", "Bonjour, comment allez-vous", "hello"]
# greeting_responses = ["hey", "hey how are you?", "*nods*", "hello, how you doing", "hello", "Welcome, I am good and you"]

byeAnswer = ['bye','goodbye']



if "history" not in st.session_state:
    st.session_state.history = []

st.title("Project Chatbot : j'recommande ton resto")

def generate_answer():
    human_text = st.session_state.input_text
    human_text_lowered = human_text.lower()
    if human_text_lowered not in byeAnswer:
        if human_text_lowered in greeting_inputs:
            st.session_state.history.append({"message": human_text, "is_user": True,"key":random.randint(1,100000)})
            st.session_state.history.append({"message": random.choice(greeting_responses), "is_user": False,"key":random.randint(1,100000)})
        else:
            st.session_state.history.append({"message": human_text, "is_user": True,"key":random.randint(1,100000)})
            if human_text_lowered == 'merci' or human_text_lowered == 'merci beaucoup':
                st.session_state.history.append({"message": 'De rien !', "is_user": False,"key":random.randint(1,100000)})
            else:
                st.session_state.history.append({"message": generate_response(human_text_lowered), "is_user": False,"key":random.randint(1,100000)})
    else:
        st.session_state.history.append({"message": human_text, "is_user": True,"key":random.randint(1,100000)})
        st.session_state.history.append({"message": "Aurevoir et à bientôt", "is_user": False,"key":random.randint(1,100000)})


st.text_input("Bonjour, marre de la routine ? Vous souhaitez changer de restaurant ? Taper le nom d'un restaurant et je vous ferais mes meilleurs recommandations !", key="input_text", on_change=generate_answer)

# HelloText = "Hello, I am the American Airlines Chatbot. You can ask me any question regarding our company :"
# st.session_state.history.append({"message": HelloText, "is_user": False,"key":random.randint(1,100000)})

for chat in st.session_state.history:
    st_message(**chat)  # unpacking