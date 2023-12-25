import streamlit as st
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from answer_questions import answer_question

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
    name="Readwise Reading Summarizer",
    description=
    "Useful for questions about books, concepts, ideas, definitions of terms, and more. A tool that can summarize the answers to questions from a user's Readwise book, article, and paper highlights. Please provide a full question.",
    func=lambda q: str(answer_question(q)),
    return_direct=True)
]

# instantiate memory
memory = ConversationBufferMemory(memory_key="chat_history")

# initialize agent
agent_chain = initialize_agent(tools,
                               llm,
                               agent="conversational-react-description",
                               memory=memory)

