from langchain_core.output_parsers  import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import answer_examples
from langchain_core.runnables import RunnableLambda

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=149)
    index_name = 'tax-index-markdown'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever=database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
   
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    return history_aware_retriever

def get_llm(model="gpt-4o"):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        거주자라는 말은 반드시 들어가야합니다. 다만 이미 거주자라고 표현되어 있는 경우 질문만 리턴해주세요.
        사전: {dictionary}

        질문: {{question}}
    """)
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain

def get_rag_chain():
    llm = get_llm()
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활요해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "다만 소득세에 관련된 내용이지만 문서에 제공하고 있지 않다면 개념 설명만 해주세요."
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장 정도의 짧은 내용의 답변을 원합니다."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain

def route(info):
    topic = info["topic"].lower()

    if topic == "tax":
        return default_chain()  # RAG로 답변
    if topic == "greeting":
        return "안녕하세요! 소득세 관련해서 궁금한 점을 질문해 주세요. 😊"
    return "소득세법 관련 법률에 대한 질문이 아니라서 답변할 수 없습니다."
    

def get_only_tax_chat_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """아래 질문을 보고 분류 라벨만 출력하세요.
        - 'tax': 소득세법/세법/소득세 관련 법률 질문
        - 'greeting': 짧은 인사/감사/호출(예: 안녕하세요, 하이, 고마워요, 테스트)
        - 'other': 그 외 모든 것

        반드시 tax/greeting/other 중 하나만 출력.
        질문: {question}"""
    )
    only_tax_chat_chain = prompt | llm | StrOutputParser()
    return only_tax_chat_chain

def default_chain():
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    tax_chain = {"input":dictionary_chain} | rag_chain

    return tax_chain

def get_ai_response(user_message, session_id: str):
    first_chain = get_only_tax_chat_chain()
    first_chain = first_chain.invoke({"question": user_message})

    full_chain = {
        "topic": lambda _: first_chain, 
        "question": lambda x: x["question"],
        } | RunnableLambda(route)
    
    ai_message = full_chain.stream(
        {
            "question":user_message
        }, 
        config={
            "configurable" : {
                "session_id": session_id
            }    
        }
    )
    return ai_message



# def get_ai_message(user_message):
    
#     dictionary_chain = get_dictionary_chain()
#     rag_chain = get_rag_chain()

#     tax_chain = {"input":dictionary_chain} | rag_chain
#     ai_message = tax_chain.invoke(
#         {
#             "question":user_message
#         }, 
#         config={
#             "configurable" : {
#                 "session_id": "abc123"
#             }    
#         })
#     return ai_message
