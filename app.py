import streamlit as st
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from answer_questions import answer_question

title = "Advanced Langchain Agent"
title = st.markdown(
    f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True
)


# define LLM
llm = ChatOpenAI(
    temperature=0, 
    model="gpt-4", 
    max_tokens=1000, 
    openai_api_key=st.secrets["OPENAI_API_KEY"]
    )

# define tools
tools = [
  Tool(
    name="Indigenous Narratives & Opioid Crisis Analyzer",
    description=
    "Specialized for insights into the disproportionate impact of the opioid crisis on First Nations communities, the science of storytelling, and related research in Canada and America. This tool distills answers from a vast collection of PDFs and texts, focusing predominantly on First Nations narratives and opioid-related studies. Provide a comprehensive question for targeted summaries from your extensive body of research material.",
    func=lambda q: str(answer_question(q)),
    return_direct=True)
    ]

# instantiate memory
memory = ConversationBufferMemory(memory_key="chat_history")

# initialize agent
agent_chain = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    memory=memory
)

if query := st.chat_input(
    "Ask a question about the opioid crisis and First Nations communities."
):
    with st.chat_message(name="user"):
        st.write(query)

    with st.chat_message(name="assistant"):
        st.write(agent_chain(query))
