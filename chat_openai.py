from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks import get_openai_callback


chat = ChatOpenAI(temperature=0)

initial_prompt = "You are a coding assistant. When a user asks you a question you will respond with the correct answer and explain how you arrived at the answer."

messages = [
    SystemMessage(content=initial_prompt)
]

session_tokens = 0

# present an interactive prompt for the user
while True:
    user_input = input("\n~> ")
    if user_input == "exit":
        break
    if user_input:
        with get_openai_callback() as cb:
            messages.append(HumanMessage(content=user_input))
            response = chat(messages)
            messages.append(AIMessage(content=response.content))
            tokens = cb.total_tokens
        print(response.content)