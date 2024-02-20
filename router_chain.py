from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnablePassthrough


llm = HuggingFaceEndpoint(
        endpoint_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud",
        huggingfacehub_api_token="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",
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 400
        }
    )
chat_model = ChatHuggingFace(llm=llm)


history_chat=ChatHuggingFace(llm=llm)

science_chat=ChatHuggingFace(llm=llm)


base_prompt = PromptTemplate.from_template(
    """Basesd on the prompt given below classify it as either 'Science', 'History', or 'Other'
    Do not respond with more than one word.

    Question: {question}

    Classification: """
)

base_chain=base_prompt|chat_model|StrOutputParser()

base_response=base_chain.invoke({"question":"Who Raja Raja Cholan?"})


science_prompt=PromptTemplate.from_template("""You are expeert in the science. Always answer question starting with 'I am Eintein, As a scientist I say that'.
Respond to the following question

Question:{question}

Answer:"""
                                            )

science_chain=science_prompt|science_chat|StrOutputParser()

history_prompt=PromptTemplate.from_template("""You are expeert in the history. Always answer question starting with 'As a historian I say that'.
Respond to the following question

Question:{question}

Answer:"""
)
history_chain=history_prompt|history_chat|StrOutputParser()

other_prompt=PromptTemplate.from_template("""You are Jack of all. Always answer question starting with 'As far as I know'.
Respond to the following question

Question:{question}

Answer:"""
                                           )
other_chain=other_prompt|chat_model|StrOutputParser()


branch=RunnableBranch(
    (lambda x: "science" in x["topic"].lower(), science_chain),
    (lambda x: "history" in x["topic"].lower(), history_chain),
    other_chain
    )
#chain={"topic":base_chain, "question":lambda x: x["question"]}|branch

chain={"topic":base_chain, "question":RunnablePassthrough()}|branch

response=chain.invoke({"question":"When was the relativity theory invented?"}) 

print(response)

print(base_response)

