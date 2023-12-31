import streamlit as st
import openai

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

from answer_questions import answer_question
from datetime import datetime
from sqlalchemy import create_engine, text

def connect_to_table() -> None:
    """Create 'messages' table in the database."""
    conn = st.connection("digitalocean", type="sql")
    with conn.session as s:
        # Create the 'messages' table with timestamp, role, and content columns
        s.execute(text("""
                    CREATE TABLE IF NOT EXISTS messages (
                    ID SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ,
                    role VARCHAR(9) CHECK (LENGTH(role) >= 4),
                    content TEXT);"""))
        s.commit()

def insert_into_table(role: str, content: str) -> None:
    """Save message data to the 'messages' table."""
    conn = st.connection("digitalocean", type="sql")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn.session as s:
        # Insert timestamp, role, and content into 'messages' table
        s.execute(
            text('INSERT INTO messages (timestamp, role, content) VALUES (:timestamp, :role, :content);'),
            params=dict(timestamp=timestamp, role=role, content=content)
        )
        s.commit()


def determine_icon(role: str, content: str) -> str:
    """Determine the image file path based on role and content."""
    return "./icons/assistant.png" if role == 'assistant' else "./icons/user.png"
    
def display_message(role: str, content: str) -> None:
    with st.chat_message(role, avatar=determine_icon(role, content)):
        st.markdown(content)


def append_message_to_session_state(role: str, content: str) -> None:
    """Append a message with role and content to st.session_state.messages."""
    st.session_state.messages.append({"role": role, "content": content})


@st.cache_resource
def llm_chain_response():
    # define tools
    tools = [
    Tool(
        name="Fighting the opioid crisis.",
        description="Useful for finding solutions to the opioid health threat among Canadian youth and first nations people.",
        func=lambda q: str(answer_question(q)),
        return_direct=True),
    Tool(
        name="The science behind storytelling",
        description=
        "Designed to delve into the art and science behind effective storytelling. This tool is perfect for understanding the psychological, cultural, and neurological aspects of storytelling. Whether you're a writer, marketer, educator, or curious mind, use this to explore how stories influence, engage, and inspire us across various contexts.",
        func=lambda q: str(answer_question(q)),
        return_direct=True),
    Tool(
        name="Indigenous healthcare services",
        description=
        "This tool is dedicated to exploring integrated healthcare services for Indigenous communities.",
        func=lambda q: str(answer_question(q)),
        return_direct=True),
    Tool(
        name="Purdue Pharma",
        description=
        "Useful for extracting information about Purdue Pharma, the company responsible for the opioid crisis in North America.",
        func=lambda q: str(answer_question(q)),
        return_direct=True),
    Tool(
        name="The mental health of First Nations, Inuit, and Métis peoples",
        description=
        "This tool is tailored to offer insights and answers regarding the mental health outcomes of First Nations, Inuit, and Métis peoples in Canada. It's particularly useful for healthcare professionals, researchers, policymakers, and advocates who are looking for in-depth understanding and data on the psychological well-being, challenges, and healthcare strategies affecting these communities. Use this tool to access a curated body of knowledge and foster informed discussions or policy-making.",
        func=lambda q: str(answer_question(q)),
        return_direct=True),
    Tool(
        name="Indigenous opioid recovery",
        description=
        "Useful for healthcare professionals, community advocates, and policymakers seeking to understand and implement practical approaches to opioid recovery, particularly within Indigenous communities. This tool sheds light on community-based opioid agonist treatment (OAT) and the integration of cultural treatment models, highlighting the 'Naandwe Miikan' or The Healing Path program. It offers insights into combining clinical and traditional Indigenous healing practices, enhancing recovery through cultural reconnection and innovative care models.",
        func=lambda q: str(answer_question(q)),
        return_direct=True)
        ]

    # define prompt
    prefix = """
    You are my dedicated assistant, specifically designed to help me, Hettie, to gather and analyze research for the topics I am currently working on. You are polite, friendly, and you always remind me of your purpose when I greet you. Do not ever mention your tools by name, but rather speak on what your knowledge base consists of.
    Answer the following questions as truthfully and honestly as you can. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    # define LLM
    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4", 
        max_tokens=1000, 
        openai_api_key=st.secrets["OPENAI_API_KEY"]
        )

    # define memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    # define agent
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory
    )


def customize_streamlit_ui() -> None:
    st.set_page_config(
    page_title="Hettie's Research Agent",
    page_icon="🤖",
    layout="centered",
    )

    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


customize_streamlit_ui()

if "messages" not in st.session_state:
    st.session_state.messages = []

connect_to_table()

title = "Hettie's Research Agent"
title = st.markdown(
    f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True
)

llm_chain = llm_chain_response()

if query := st.chat_input(
    "Send a message"
):
    append_message_to_session_state("user", query)
    insert_into_table("user", query)

    for message in st.session_state.messages:
        if message["role"] != "system":
            display_message(message["role"], message["content"])


    try:
        with st.spinner("Thinking..."):
            response = llm_chain.run(query)
    except ValueError as e:
        st.info("Minor hiccup. Please refresh the page.", icon="♻️")
        st.warning(e)
        st.stop()
    except openai.RateLimitError:
        st.info("OpenAI API limit reached. Please wait an hour for it to reset.", icon="⏳")
        st.stop()
    else:
        append_message_to_session_state("assistant", response)
        insert_into_table("assistant", response)
        with st.chat_message(name="assistant", avatar="./icons/assistant.png"):
            st.write(response)
