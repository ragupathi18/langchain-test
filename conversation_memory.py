from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#memory.chat_memory.add_user_message("Hi!, I want know about Canada")
#memory.chat_memory.add_ai_message("Sure?")

chat_history=memory.load_memory_variables({})

#print(chat_history)
                                    



prompt = PromptTemplate.from_template(
    """You are current affairs expert. Answer the question precisely. Do not add aditional question and answer messages.
    Previous Conversation:{chat_history}

    New Human Question: {question}
    Response: 
    """
)
llm = HuggingFaceEndpoint(
        endpoint_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud",
        huggingfacehub_api_token=****,
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
response=llm_chain({"question":"Where is United Kingdom?"})

print(response)

response=llm_chain({"question":"Who is the Prime Minister?"})

print(response)

