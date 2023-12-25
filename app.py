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


if "messages" not in st.session_state:
    st.session_state.messages = []


title = "Advanced Langchain Agent"
title = st.markdown(
    f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True
)

st.write("Provide me with overdose statistics related to first nations people in Canada.")

@st.cache_resource
def llm_chain_response():
    # define tools
    tools = [
    Tool(
        name="Indigenous Narratives & Opioid Crisis Analyzer",
        description=
        "Useful for answering questions about the opioid crisis and First Nations communities.",
        func=lambda q: str(answer_question(q)),
        return_direct=True)
        ]

    # define prompt
    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
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
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    return agent_chain


# instantiate memory
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# conversation = LLMChain(
    # llm=llm,
    # prompt=prompt,
    # verbose=True,
    # memory=memory
# )
# memory = ConversationBufferMemory(memory_key="chat_history")




# initialize agent
# agent_chain = initialize_agent(
#     tools,
#     llm,
#     agent="conversational-react-description",
#     memory=memory,
#     verbose=True,
#     prompt=prompt
# )


if query := st.chat_input(
    "Ask a question about the opioid crisis and First Nations communities."
):
    append_message_to_session_state("user", query)

    # with st.chat_message(name="user"):
    #     if query not in st.session_state.messages:
    #         st.write(query)

    for message in st.session_state.messages:
        if message["role"] != "system":
            display_message(message["role"], message["content"])

    llm_chain = llm_chain_response()
    response = llm_chain.run(query)
    # response = agent_chain(query)
    append_message_to_session_state("assistant", response)

    with st.chat_message(name="assistant"):
        # st.write(response["output"])
        st.write(response)
        # st.write(response["agent_scratchpad"])
