from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain


prompt = PromptTemplate.from_template(
    """You are current affairs expert. Answer the question precisely. Do not add aditional question and answer messages.
    Previous Conversation:{chat_history}

    New Human Question: {question}
    Response: 
    """
)
llm = HuggingFaceEndpoint(
        endpoint_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud",
        huggingfacehub_api_token="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",
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )


memory=ConversationSummaryMemory(llm=llm)


conversation_with_summary=ConversationChain(llm=llm,
                                           memory=memory,
                                           verbose=True)

conversation_with_summary.predict(input="Hello, My member id is MEM1234R5")

conversation_with_summary.predict(input="I visited Doctor Mr RIO on Jan 5 for colonoscopy")

conversation_with_summary.predict(input="What is the status of my claim?")
