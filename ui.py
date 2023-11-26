from lotr1 import lotr

import streamlit as st




st.markdown("<h1 style='color:lightgreen;'>KPMG Data Harbor 📊</h1>", unsafe_allow_html=True)

st.image(r"/Users/nirbhaysedha/Desktop/RAG/kpmg-logo-1.webp", width=100)

query=st.text_input("enter the query for pdf")

# queru=" Explain Chapter 1 -- An Overview of Financial Management "
docs=lotr.get_relevant_documents(query)


from dotenv import load_dotenv
load_dotenv()
import os
from langchain.llms import GooglePalm 
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)





template =f"Answer this question {query} using this information={docs} , make sure you answer itefficiently and make the answer concise !"

answer=llm.predict(template)
st.write(answer)


