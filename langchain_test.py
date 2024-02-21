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
        huggingfacehub_api_token=*****,
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
                "You are a helpful assistant that re-writes the user's text to sound more uplifting"
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

print(chat_model(chat_template.format_messages(text="i do not like frogs.")))

print("\n\n for Oviya \n\n")
print(chat_model.invoke(chat_template.invoke({"text":"I am so sad because  I dont have a valentine"})))

print("\n\n for Madhiniya \n\n")
print(chat_model.invoke(chat_template.invoke({"text":"Why are you annoying?"})))
