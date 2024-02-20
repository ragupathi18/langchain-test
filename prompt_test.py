from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} story about {content}."
)
response=prompt_template.format(adjective="scary", content="ghosts")
print(response)

response=prompt_template.invoke({"adjective":"scary", "content":"ghosts"})
print(response)


print("Chat Prompt Template below....")
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
        ("assistant", "I am sick"),
        ("user", "Get well soon"),
    ]
)




messages = chat_template.format_messages(name="Jake", user_input="What is your name?")

print(messages)

response=chat_template.invoke({"name":"Albert","user_input":"Where are your in the world"})


print(response)
