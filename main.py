import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,classification_report
def clean_data(text):
    text=str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def token_overlab(a,b):
    set1=set(a.split())
    set2=set(b.split())

    overlap =len(set1.intersection(set2))
    return overlap


data = pd.read_excel('excels/train_dataset.xlsx')

data['category_a'] = data['category_a'].apply(clean_data)
data['category_b'] = data['category_b'].apply(clean_data)

combined_text= pd.concat([
    data['category_a'],
    data['category_b']
])

# Finding n-gram Word Similiarity 
vectorize = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),
    min_df=1
)


vectorize.fit(combined_text)

category_a_vector = vectorize.transform(data['category_a'])
category_b_vector = vectorize.transform(data['category_b'])


scores=[]

for i in range(len(data)):
    score = cosine_similarity(
        category_a_vector[i],
        category_b_vector[i]
    )[0][0]
    scores.append(score)


data['similiarity']=scores


# Finding n-gram Character Similiarity 

char_vectorizer=TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,5),
    min_df=1
)

char_vectorizer.fit(combined_text)

vec_a=char_vectorizer.transform(data['category_a'])
vec_b=char_vectorizer.transform(data['category_b'])


char_scores=[]
for i in range(len(data)):
    score=cosine_similarity(
        vec_a[i],
        vec_b[i]
    )[0][0]
    char_scores.append(score)

data['char_similiarity']=char_scores


data["length_diff"] = abs(
    data["category_a"].str.len() -
    data["category_b"].str.len()
)

data['token_overlap']=data.apply(
    lambda x:token_overlab(
        x['category_a'],
        x['category_b']
    ),
    axis=1
)


x=data[[
    'similiarity',
    'char_similiarity',
    'length_diff',
    'token_overlap'
]]
y=data['label']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = SGDClassifier(
    loss='log_loss',
    max_iter=1000,
    learning_rate='constant',
    # eta0=0.01,
    # warm_start=True,
    random_state=42    
)

epochs =20 

for epoch in range(epochs):
    model.fit(x_train,y_train)
    predict = model.predict(x_test)
    accuracy = accuracy_score(
        y_test,predict
    )
    print(f"Epoch {epoch+1} | Accuracy: {accuracy}")
    print(classification_report(y_test,predict)   )