import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


dotenv_path=find_dotenv()
load_dotenv(dotenv_path)

model=SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index("ai-assistant")


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, include_metadata=True)

    # Check if there are matches
    if len(result['matches']) == 0:
        return "No matches found."

    # Prepare the text output based on the number of matches
    text_output = ""
    for match in result['matches']:
        if 'metadata' in match and 'text' in match['metadata']:
            text_output += match['metadata']['text'] + "\n"

    return text_output.strip()  # Remove trailing newline


# def find_match(input):
#     input_em=model.encode(input).tolist()
#     result=index.query(vector=input_em,top_k=2,include_metadata=True)
#     return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def query_refiner(conversation, query):
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    prompt = f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=256, num_return_sequences=1, no_repeat_ngram_size=2)
    refined_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    refined_query = refined_query.split("Refined Query:")[-1].strip()
    return refined_query

def get_conversation_string():
    conversation_string=""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string+="Human: "+st.session_state['requests'][i] + "\n"
        conversation_string+="Bot: "+st.session_state['responses'][i+1] + "\n"
    return conversation_string