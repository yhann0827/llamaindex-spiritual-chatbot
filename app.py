import openai
import json
import os
import pandas as pd
import streamlit as st
from bert_score import score
from llama_index.core import VectorStoreIndex, Document, Settings  # Updated imports
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding  # For embedding model
from llama_index.core.prompts import PromptTemplate  # Updated prompt import
from llama_index.core.chat_engine.types import ChatMode  #
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever

data=[]

OpenAI.api_key = st.secrets["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

def detect_emotion(user_input):
    prompt = f"""請根據以下文字判斷情緒，可以返回多個情緒名稱（擔心, 恐懼, 解脫, 自責, 傷心, 愛, 面對, 繼承, 再見, 懺悔, 寬恕, 放下, 罪惡感, 隱瞞, 合一, 封閉, 請讓我走, 原諒, 思念, 愛, 永恆, 
            孤單, 失落, 存在, 思念, 靈魂連結, 溝通, 溝通, 罣礙, 怨念, 願望, 圓滿, 改變, 執著, 憂心, 愛的延續, 情執, 前世今生, 角色扮演, 輪迴, 轉世, 告別, 感恩, 前世有約, 愛的承諾, 緣起緣滅, 光, 天堂, 榮耀,
            懺悔與感悟, 感恩與贈禮, 安慰與歸屬, 圓滿與祝福, 光明與愛, 解脫與無所不在），若適用請以逗號分隔：
            {user_input}"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role":"assistant", "content":"你是一個能分析情緒的助手"},
            {"role":"user", "content":prompt}]
    )
    emotions = [emotion.strip() for emotion in response.choices[0].message.content.split(',')]
    print("Detected Emotions:", emotions)
    return emotions

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner("Loading messages..."):
        with open("output.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)

        docs = [Document(
            text=entry["訊息"],
            metadata={"emotion": entry["情緒"], "title": entry["標題"]}) for entry in json_data]
        # service_context = ServiceContext.from_defaults(
        #     llm=OpenAI(model="gpt-4", temperature="0.1", system_prompt="你是一位来自来世的慰藉引导者，以爱与智慧与人交谈，并根据人们的情绪进行交流。")
        # )
        index=VectorStoreIndex.from_documents(docs)
        return index

def retrieve_message_from_dataset(index, emotion):
    if isinstance(emotion, str):
        query = emotion.strip()
        emotion_list = [emotion.strip()]
    elif isinstance(emotion, list):
        query = " ".join([e.strip() for e in emotion])  
        emotion_list = [e.strip() for e in emotion]
    else:
        query = "根据上述用户的情绪，给他适当的慰籍和开导。" 
        emotion_list = []
    
    # Retrieve nodes using the emotion-based query
    bm25_retriever = BM25Retriever.from_defaults(index, similarity_top_k=5)

    nodes = bm25_retriever.retrieve(query)
    print("Retrieved Nodes:", [(node.text, node.metadata) for node in nodes])
    matching_nodes = [node for node in nodes if node.metadata["emotion"] in emotion_list]
    print("Matching Nodes:", [(node.text, node.metadata) for node in matching_nodes])
    
    return [(node.text, node.metadata["title"]) for node in matching_nodes]

# def evaluate_faithfulness(response, contexts):
#     if isinstance(response, AgentChatResponse):
#         response = response.response  # Extract string content

#     if response is None or response == "":
#         raise ValueError("Response must be a non-empty string.")
    
#     if isinstance(contexts, str):  
#         contexts = [contexts]  # Convert a single string into a list
    
#     print(f"Response Type: {type(response)}, Value: {response}")
#     print(f"Contexts Type: {type(contexts)}, Value: {contexts}")

#     evaluator = FaithfulnessEvaluator()

#     try:
#         evaluation_score = evaluator.evaluate(response=response, contexts=contexts)
#         print("Score:", evaluation_score)
#         return evaluation_score
#     except ValueError as e:
#         print(f"Evaluation failed: {e}")
#         raise  # Re-raise the error for debugging

def evaluate_response(response, contexts):
    if isinstance(response, AgentChatResponse):
        response = response.response

    if response is None or response == "":
        raise ValueError("Response must be a non-empty string.")
    
    if isinstance(contexts, str):  
        contexts = [contexts] 

    P, R, F1 = score([response], contexts, lang="zh", rescale_with_baseline=True)
    
    scores = {
        "Precision": P.item(),
        "Recall": R.item(),
        "F1 Score": F1.item(),
        "Faithfulness": F1.item(),  # Faithfulness (similar to F1)
        "Relevance": R.item(),  # Recall represents how much of the reference is covered
        "Coherence": P.item(),  # Precision represents how relevant the response is to the reference
        "Fluency": F1.item()  # F1 often aligns with fluency in BERTScore
    }

    print("Evaluation Scores:", scores)
    return scores

def generate_response(index, user_input, model, temperature, max_tokens, top_p, frequency_penalty):
    detected_emotion = detect_emotion(user_input)
    messages=retrieve_message_from_dataset(index, detected_emotion)
    context_str='\n'.join([f"{title}:{text}" for text, title in messages]) if messages else "No messages found"
    chat_prompt = PromptTemplate(
        "你是一位來自往生世界的嚮導，請幫助使用者理解這些訊息。\n\n"
        "請直接以 **第一人稱** 來說話，確保語氣自然且情感真摯。\n"
        "請將以下訊息融合成一段流暢且自然的回應，\n"
        "確保不提及具體的情緒分類名稱，只讓內容傳遞出應有的情感。\n"
        "請確保完整保留所有內容，不要省略、總結、或加入額外解釋。\n\n"
        "===請用這些內容來回應使用者===\n"
        "{context_str}\n"
        "===請根據上述內容，自然地表達你的回應==="
    )

    print("context str", context_str)
    llm = OpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty
    )
    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        system_prompt=chat_prompt.format(context_str=context_str),
        llm=llm
    )
    response = chat_engine.chat(user_input)
    evaluation_scores=evaluate_response(response=response, contexts=context_str)
    data.append({
        "Query": user_input,
        "Response": response,
        "Context": context_str,
        "Faithfulness": evaluation_scores["Faithfulness"],
        "Relevance": evaluation_scores["Relevance"],
        "Coherence": evaluation_scores["Coherence"],
        "Fluency": evaluation_scores["Fluency"]
    })
    df=pd.DataFrame(data)
    df.to_csv("evaluation.csv", mode='a', index=False, header=not os.path.exists("evaluation.csv"))
    return response

def main():
    st.set_page_config(page_title="Spirit Chatbot", page_icon="🕊️")
    index=load_data()

    st.header("🕊️ Spirit Chatbot")

    if st.sidebar.button("Reset Chat"):
        st.session_state["messages"]=[{"role":"assistant", "content":"你好！有什么我可以帮助你的吗？"}]
        model = st.sidebar.selectbox("Choose AI Model", ["gpt-4.5-preview", "gpt-4o", "chatgpt-4o-latest"])

    model = st.sidebar.selectbox("Choose AI Model", ["gpt-4o", "chatgpt-4o-latest"])
    st.sidebar.markdown("### ⚙️Model Parameters")
    temperature = st.sidebar.number_input("Temperature (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    max_tokens = st.sidebar.number_input("Max Tokens (50 - 4096)", min_value=50, max_value=4096, value=500, step=10)
    top_p = st.sidebar.number_input("Top-p (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    frequency_penalty = st.sidebar.number_input("Frequency Penalty (-2.0 to 2.0)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    
    if "messages" not in st.session_state:
        st.session_state["messages"]=[{"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_input("告訴我你的感受..."):
        st.session_state.messages.append({'role':'user', 'content':user_input})
        st.chat_message('user').markdown(user_input)
    
        response = generate_response(index, user_input,  model, temperature, max_tokens, top_p, frequency_penalty)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role':'assistant', 'content':response})
        

if __name__=="__main__":
    main()