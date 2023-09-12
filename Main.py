from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

# from langchain.schema import (
#     SystemMessage,
#     HumanMessage,
#     AIMessage
# )


def main():

    load_dotenv()
    # print("MY api key is: ", os.getenv('OPENAI_API_KEY'))


    # test our api key
    if os.getenv('OPENAI_API_KEY') is None or os.getenv("OPENAI_API_KEY")=="":
        print("Invalid or empty string in OPENAI_KEY")
        exit(1)
    else:
        print("ALL GOOD")

    # chat = ChatOpenAI(temperature=0.9)
    llm = ChatOpenAI()
    conversation = ConversationChain(llm=llm,
                                     memory=ConversationEntityMemory(llm=llm),
                                     prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,

    #for summerrising or generating the info

                                     verbose=True
                                     )



    # messages = [
    #     SystemMessage(content="Your a  helpful assistant"),
    # ]
    print("Hello , I am chatgpt-cli")
    # using buffffer memeory
    while True:
        user_input = input(">")
        ai_response = conversation.predict(input=user_input)
        print("\n Assistant :", ai_response)



    # basic loop
    # while True:
    #     user_input = input(">")
    #     messages.append(HumanMessage(content=user_input))
    #
    #     ai_response=  chat(messages)
    #     messages.append(AIMessage(content=ai_response.content))
    #     print("\n Assistant says", ai_response.content)
    #
    #
    #     # print hostoiry of messgaaes
    #     print("history ",messages)
    #
    #  using conversation buffer memory




# memory tutorial
# conversationalbuffer
# entity memory

if __name__ == '__main__':
    main()


