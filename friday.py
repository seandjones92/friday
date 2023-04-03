#!/usr/bin/env python3

from langchain.agents import load_tools
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.callbacks import get_openai_callback

tools = load_tools(["wikipedia", "wolfram-alpha", "human"])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0.0)
agent_chain = initialize_agent(tools, llm, memory=memory, agent="chat-conversational-react-description", verbose=False)

print('Enter your question or "exit" to quit.')
while True:
    user_input = input("\n~> ")
    if user_input == "exit":
        break
    if user_input:
        with get_openai_callback() as cb:
           results = agent_chain.run(input=user_input)
           print(results)
           cost = str(round(cb.total_cost, 2))
           print("\nTotal cost of session: $" + cost)
