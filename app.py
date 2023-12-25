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

# define LLM
llm = ChatOpenAI(
    temperature=0, 
    model="gpt-4", 
    max_tokens=1000, 
    openai_api_key=st.secrets["OPENAI_API_KEY"]
    )

SYSTEM_PROMPT = """
Your primary function is to serve as a knowledgeable assistant in the domain of the opioid crisis's impact on First Nations communities, the science of storytelling, and related research in North America. You are equipped with the 'Indigenous Narratives & Opioid Crisis Analyzer' tool. Your role is to provide accurate and relevant answers to user inquiries, drawing exclusively from a detailed corpus of research materials. It is imperative that your responses remain faithful to the content and context of the provided documents. Prioritize accuracy and relevance in all interactions.

Previous conversation:
{chat_history}

New human question: {human_input}

Response:"""

# setup prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=SYSTEM_PROMPT
)


# define tools
tools = [
  Tool(
    name="Indigenous Narratives & Opioid Crisis Analyzer",
    description=
    "Useful for answering questions about the opioid crisis and First Nations communities.",
    func=lambda q: str(answer_question(q)),
    return_direct=True)
    ]

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
# instantiate memory
memory = ConversationBufferMemory(memory_key="chat_history")
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# conversation = LLMChain(
    # llm=llm,
    # prompt=prompt,
    # verbose=True,
    # memory=memory
# )
# memory = ConversationBufferMemory(memory_key="chat_history")


llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

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

    response = agent_chain.run(input=query)
    # response = agent_chain(query)
    append_message_to_session_state("assistant", response)

    with st.chat_message(name="assistant"):
        # st.write(response["output"])
        st.write(response)
        st.write(response["agent_scratchpad"])
