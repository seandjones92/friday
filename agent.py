#!/usr/bin/env python3

from langchain.llms import OpenAI

from langchain.agents import initialize_agent, Tool, load_tools

from langchain.callbacks import get_openai_callback

from langchain.document_loaders.sitemap import SitemapLoader

from langchain.indexes import VectorstoreIndexCreator


# initialize the llm
llm = OpenAI()

# load the tools
tools = load_tools(["wikipedia", "wolfram-alpha", "terminal", "python_repl", "llm-math", "human"], llm=llm)

# initialize the agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# track number of tokens used total
session_tokens = 0

# MOTD
print('Enter your question or "exit" to quit.')

# present an interactive prompt for the user
while True:
    user_input = input("\n~> ")
    if user_input == "exit":
        break
    if user_input:
        with get_openai_callback() as cb:
            response = agent.run(user_input)
            tokens = cb.total_tokens
        print(response)
        print("\n\n--------------------------------")
        print("Tokens used for this query: %s" % tokens)
        session_tokens += tokens
        print("------------------------------------")
        print("Total tokens used this session: %s" % session_tokens)
        print("------------------------------------")
        print("Total cost of this session: $%s" % cb.total_cost)
