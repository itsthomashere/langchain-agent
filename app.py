import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

from answer_questions import answer_question


def display_message(role: str, content: str) -> None:
    with st.chat_message(role):
        st.markdown(content)


def append_message_to_session_state(role: str, content: str) -> None:
    """Append a message with role and content to st.session_state.messages."""
    st.session_state.messages.append({"role": role, "content": content})


@st.cache_resource
def llm_chain_response():
    # define tools
    tools = [
    Tool(
        name="Indigenous Narratives & Opioid Crisis Analyzer",
        description=
        "Useful for answering questions about the opioid crisis and First Nations communities.",
        func=lambda q: str(answer_question(q)),
        return_direct=True),
    Tool(
        name="Science of Storytelling Explorer",
        description=
        "Designed to delve into the art and science behind effective storytelling. This tool is perfect for understanding the psychological, cultural, and neurological aspects of storytelling. Whether you're a writer, marketer, educator, or curious mind, use this to explore how stories influence, engage, and inspire us across various contexts.",
        func=lambda q: str(answer_question(q)),
        return_direct=True)
        ]

    # define prompt
    prefix = """
    You are my dedicated assistant, specifically designed to help me, Hettie, to gather and analyze research for the topics I am currently working on. You are polite, friendly, and you always remind me of your purpose when I greet you.
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
        agent=agent, tools=tools, verbose=True, memory=memory
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

title = "Advanced Langchain Agent"
title = st.markdown(
    f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True
)


if query := st.chat_input(
    "Ask a question about the opioid crisis and First Nations communities."
):
    append_message_to_session_state("user", query)

    for message in st.session_state.messages:
        if message["role"] != "system":
            display_message(message["role"], message["content"])

    llm_chain = llm_chain_response()
    response = llm_chain.run(query)
    append_message_to_session_state("assistant", response)

    with st.chat_message(name="assistant"):
        st.write(response)
