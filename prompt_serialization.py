from langchain.prompts import load_prompt

prompt=load_prompt("serialized_prompt.json")

print(prompt.format(adjective="scary", content="ghosts"))

prompt.save("test.json")

      
