from scipy.spatial.distance import cosine
import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score, dcg_score

df=pd.read_csv('data_new.csv')
features_df=pd.read_csv("features.csv")
index_df=pd.read_csv('index.csv')

min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features_df) 
count_vec = CountVectorizer(stop_words='english')
count_matrix = count_vec.fit_transform(index_df['feature'])
cosine_sim = cosine_similarity(count_matrix, count_matrix) 

def knn(m):
  model = neighbors.NearestNeighbors(n_neighbors=m, algorithm='auto')
  model.fit(features) 
  dist, idlist = model.kneighbors(features)
  return dist,idlist 

def Recommender(name,n):
    dist,idlist=knn(n)
    prod_list = []
    id_list=[]
    recomm=[]
    prod_id = df[df['title'] == name].index
    prod_id = prod_id[0]
    for newid in idlist[prod_id]:
        prod_list.append(df.iloc[newid].title)
        id_list.append(newid)
    for i in range(0,len(prod_list)):
      recomm.append(f"{prod_list[i]}")
    return recomm

def cosine_recommender(nam:str,n):
    ind = index_df[index_df['title'] == nam].index.to_list()[0]
    cos_scor = list(enumerate(cosine_sim[ind]))
    cos_scor = sorted(cos_scor, key=lambda x: x[1], reverse=True)
    cos_scor = cos_scor[0:n]
    new_ind = [i[0] for i in cos_scor]
    return index_df['title'].iloc[new_ind]

def combiner(prod_name,n):
  prod=list() 
  cos=cosine_recommender(prod_name,50)
  cos=cos.to_list()
  neir=Recommender(prod_name,50)
  for j in range(0,len(neir)):
    for k in range(0,len(cos)):
      if neir[j]==cos[k]:
        prod.append(neir[j]) 
  prod=prod[0:n]
  prod=list(prod)
  return prod
st.title('Ezmall Recommendation system')
input=st.selectbox('Enter the Product name:',df['title']) 
int_val = st.number_input('Number of recommendation:', min_value=5, max_value=18, value=5, step=1)
st.caption('The Input number should be between 5 and 18.')
model=st.radio("Select a model:",('Unsupervised-nearest neighbors','cosine similarity','Using Combiner'))
int_val=int(int_val)
int_val=int_val+1

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
    color:blue;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">OUTPUT:</p>', unsafe_allow_html=True)

if st.button("Show Recommendation"):
  if model=='Unsupervised-nearest neighbors': 
      st.write("Selected Model is Unsupervised-nearest neighbors") 
      knn_output=Recommender(input,int_val)
      knn_output=list(knn_output)   
      for i in range(0,len(knn_output)):
        if str(knn_output[0])==str(input):
          try:
            knn_output1=knn_output[1:]
            inp_title = df[df['title'] == knn_output[i+1]].link
            inp_category = df[df['title'] == knn_output[i+1]].custom_label_0
            inp_img = df[df['title'] == knn_output[i+1]].image_link
            inp_img=list(inp_img)
            inp_category=list(inp_category)
            url=list(inp_title)
            link1=f"[Product Page]({url[0]})"
            with st.expander(f'{i+1}) {knn_output1[i]}'):
              st.image(inp_img)
              st.markdown(link1, unsafe_allow_html=True)
              st.write(f"Category of the product: {inp_category[0]}")
          except IndexError:
            continue
        else:
          link1=f"[Product Page]({url[0]})"
          inp_title = df[df['title'] == knn_output[i]].link
          inp_category = df[df['title'] == knn_output[i]].custom_label_0
          inp_img = df[df['title'] == knn_output1[i]].image_link
          inp_category=list(inp_category) 
          inp_img=list(inp_img)
          url=list(inp_title)
          with st.expander(f'{i+1}) {knn_output[i]}'):
            st.image(inp_img)
            st.markdown(link1, unsafe_allow_html=True)
            st.write(f"Category of the product: {inp_category[0]}")
  elif model=='cosine similarity':
      st.write("Selected Model is Cosine Similarity")
      cosine_output=cosine_recommender(input,int_val)
      cosine_output=list(cosine_output)
      for i in range(0,len(cosine_output)):
          if str(cosine_output[0])==str(input):
            try:
              knn_output1=cosine_output[1:]
              inp_title = df[df['title'] == cosine_output[i+1]].link
              inp_category = df[df['title'] == cosine_output[i+1]].custom_label_0
              inp_img = df[df['title'] == cosine_output[i+1]].image_link
              inp_category=list(inp_category) 
              inp_img=list(inp_img)
              url=list(inp_title)
              link1=f"[Product Page]({url[0]})"
              with st.expander(f'{i+1}) {knn_output1[i]}'):
                st.image(inp_img)
                st.markdown(link1, unsafe_allow_html=True)
                st.write(f"Category of the product: {inp_category[0]}")
            except IndexError:
              continue
          else:
            inp_title = df[df['title'] == cosine_output[i]].link 
            inp_category = df[df['title'] == cosine_output[i]].custom_label_0
            inp_img = df[df['title'] == cosine_output[i]].image_link
            inp_category=list(inp_category) 
            inp_img=list(inp_img)
            url=list(inp_title)
            link1=f"[Product Page]({url[0]})"
            with st.expander(f'{i+1}) {cosine_output[i+1]}'):
              st.image(inp_img)
              st.markdown(link1, unsafe_allow_html=True)
              st.write(f"Category of the product: {inp_category[0]}")
  else: 
      st.write("Selected Model is Combiner") 
      combiner_output=combiner(input,int_val)
      for i in range(0,len(combiner_output)):
        if str(combiner_output[0])==str(input):
          try:
            knn_output1=combiner_output[1:]
            inp_title = df[df['title'] == combiner_output[i+1]].link
            inp_category = df[df['title'] == combiner_output[i+1]].custom_label_0 
            inp_img = df[df['title'] == combiner_output[i+1]].image_link
            inp_category=list(inp_category)
            inp_img=list(inp_img)
            url=list(inp_title)
            link1=f"[Product Page]({url[0]})"
            with st.expander(f'{i+1}) {knn_output1[i]}'):
              st.image(inp_img)
              st.markdown(link1, unsafe_allow_html=True)
              st.write(f"Category of the product: {inp_category[0]}")
          except IndexError:
            continue
        else:
          st.write(f"{i+1}) {combiner_output[i+1]}")
          inp_title = df[df['title'] == combiner_output[i]].link
          inp_category = df[df['title'] == combiner_output[i]].custom_label_0
          inp_img = df[df['title'] == combiner_output[i]].image_link
          inp_category=list(inp_category) 
          url=list(inp_title)
          inp_img=list(inp_img)
          link1=f"[Product Page]({url[0]})"
          with st.expander(f'{i+1}) {combiner_output[i+1]}'):
            st.image(inp_img)
            st.markdown(link1, unsafe_allow_html=True)
            st.write(f"Category of the product: {inp_category[0]}")

