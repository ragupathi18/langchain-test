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
            "max_new_tokens": 400
        }
    )
chat_model = ChatHuggingFace(llm=llm)


receipe_prompt=PromptTemplate.from_template("""You are a chef. Given the title of a dish, it is your job to write a recipe for that dish.
Name: {name}
Chef: This is a recipe for the above dish:""")

review_prompt=PromptTemplate.from_template( """You are a food critic for a magazine. Given the recipe of a dish, it is your job to write a review for that dish.

Dish Synopsis:
{synopsis}
Review from a food critic of the above recipe:""")

receipe_chain=receipe_prompt | chat_model|StrOutputParser()

receipe_response= receipe_chain.invoke({"name":"shrimp gumbo"})
print(receipe_response)

review_chain= {"synopsis":receipe_chain}| review_prompt|chat_model|StrOutputParser()
print("Sequential chain response below")
response=review_chain.invoke({"name":"shrimp gumbo"})
print(response)


             
