from urllib.parse import urlparse
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback
import requests
from bs4 import BeautifulSoup

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



def init_page():
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🍣"
    )
    st.header("My Great ChatGPT 🍣")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-0613"
    if model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k-0613"
    else:
        # https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4
        # 20230817時点では$1以上の課金してないので使えない
        st.session_state.model_name = "gpt-4"

    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.01とする
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    # 300: 本文以外の指示のtoken数 (決めうちの雑な実装です…)
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url


def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost


def get_content(url):
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # main, articleタグあればそれ使う
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write('something wrong')
        return None


def build_prompt(content, n_chars=300):
    return f"""以下はとあるWebページのコンテンツである。内容を{n_chars}程度でわかりやすく要約してください

=======

{content[:1000]}

=======

日本語で書いてね！
"""

def get_document(url):
    with st.spinner("Fetching Content ..."):
        loader = YoutubeLoader.from_youtube_url(
            url, 
            add_video_info=True,
            language=['en', 'ja']
        )
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.model_name,
            chunk_size = st.session_state.max_token,
            chunk_overlap=0
        )
        return loader.load_and_split(text_splitter=text_splitter) # Document取得

def summarize(llm, docs):
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.
    
{text}

ここから日本語で書いてね。必ず3段落以内の200文字以内で簡潔にまとめること:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,  # e.g. ChatOpenAI(temperature=0)
            chain_type="map_reduce",
            verbose=True,
            map_prompt=PROMPT,
            combine_prompt=PROMPT
        )
        response = chain({"input_documents": docs, "token_max": st.session_state.max_token}, return_only_outputs=True)
    return response['output_text'], cb.total_cost

def main():
    init_page()

    llm = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)

        if not is_valid_url:
            st.write('Please input valid url')
            output_text=None
        else:
            document = get_document(url)

            if document:
                with st.spinner("ChatGPT is typing ..."):
                    output_text, cost = summarize(llm, document)
                st.session_state.costs.append(cost)
            else:
                output_text = None

    if output_text:
        with response_container:
            st.markdown("## Summary")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in reversed(costs):
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    main()
