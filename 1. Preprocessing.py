import pandas as pd
import spacy

nlp = spacy.load("en_core_web_lg")

# Text cleanup

def cleanFormats(text):
    return text.str.replace("\.((jpg)|(png)|(mp3)|(mp4)|(com)|(gif)|(txt)|(doc)|(mov)|(avi))"," ", regex = True)

def clean_spec_char(text):     
    return text.str.replace("[~_]", " ", regex = True)

def cleanSeparators(text):
    return text.str.replace("|", " ", regex = False)

def cleaningText(text):
    text = text.str.lower()
    text = clean_spec_char(text)
    return  cleanSeparators(text)


# Lemmatizing and tokenizing without unrelevant data
def lemmatize(post):  
    doc = nlp(post)    
    lemmas = pd.Series([token.lemma_ for token in doc 
                       if not
                        token.is_punct  |
                        token.is_stop |
                        token.like_num |
                        token.like_url |
                        token.is_digit |
                        token.is_oov
                       ])
    return lemmas

def lemmatizeIter(df):
    posts = []
    for index, row in df.iterrows():
        print(f"Lemmatized {index + 1} posts...")
        lemmas = lemmatize(row['cleaned_post'])
        posts.append(" ".join(lemmas))
    return posts

df = pd.read_csv('data/mbti_dataset.csv', delimiter=",")
df['cleaned_post'] = cleaningText(df["posts"])
df['lemmatized_posts'] = lemmatizeIter(df)
df = df.drop(columns=['posts','cleaned_post'])
df = df.rename(columns = {"lemmatized_posts": "posts"})

df.to_csv("data/data-preprocessed.csv", index=False)