import streamlit as st
from llm import get_ai_response
from dotenv import load_dotenv



st.set_page_config(page_title="소득세 챗봇", page_icon="🚀")

st.title("🚀 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")
load_dotenv()


if 'message_list' not in st.session_state:
    st.session_state.message_list = []\

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
if user_question := st.chat_input(placeholder="소득세에 관련된 궁긍한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
        st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("Thinking..."):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            #st.write(ai_message) -> get_ai_response의 스트림은 대응할 수 없음
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "assistant", "content": ai_message})
