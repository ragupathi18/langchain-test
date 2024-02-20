from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate


prompt = PromptTemplate.from_template(
    "What would be an ideal name for a business specializing in {product}?"
)
llm = HuggingFaceEndpoint(
        endpoint_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud",
        huggingfacehub_api_token="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",
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )
chat_model = ChatHuggingFace(llm=llm)
response=prompt.invoke({"product":"Hellooooo"})
runnable = prompt | chat_model | StrOutputParser()

print(chat_model.invoke("What would be an ideal name for a business specializing in Hellooooo?"))

print(llm.generate(["What is your name"]))

response=runnable.invoke({"product":"Hellooooo"})
print(response)




print("Chat template example with LLM")
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You're a well trained scientist who provides correct and expressive answers to scientific questions."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

runnable = chat_template | chat_model | StrOutputParser()


for chunk in runnable.stream({"text": "What was the most important scientific discovery of the 20th century?"}):
    print(chunk, end="", flush=True)

print("\n\n")    
result= runnable.batch([{"text": "What was the most important scientific discovery of the 20th century?"},{"text": "Where is Moscow?"}])

print(result)
