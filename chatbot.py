from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to display the chatbot
def show_chatbot():
    client = OpenAI()

    st.title("Stock Market Expert Chatbot")
    # Set the default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Initialize messages list if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Define the system message to set the assistant's role as a stock market expert
    system_message = {
        "role": "system",
        "content": "You are a stock market expert. Provide insights, predictions, and explanations related to stock market trends and investments. If the query is unrelated to stocks or finance, respond that you are not an expert in that area."
    }

    # Display previous messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about stock marketing or finance"):
        # Append the user message to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add the system message and user messages to the request
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[system_message] + [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                ],
                stream=True,
            )
            # Collect and display the assistant's response from the stream
            response = st.write_stream(stream)

        # Add the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})


# import os
# from openai import OpenAI
# from dotenv import load_dotenv
# import streamlit as st
#
# # Load environment variables
# load_dotenv()
#
# def call_openai(text):
#     """Call OpenAI API to generate a stock-related response."""
#     if os.getenv('CHATGPT_ENABLE'):
#         try:
#             client = OpenAI()
#             completion = client.chat.completions.create(
#                 model="gpt-4",
#                 messages=[
#                     {"role": "system", "content": "You are a stock market expert. Provide insights, predictions, and explanations related to stock market trends and investments. If the query is unrelated to stocks, respond that you are not an expert in that area."},
#                     {"role": "user", "content": text}
#                 ]
#             )
#             result = completion.choices[0].message.content.strip()
#         except Exception as e:
#             result = f"An error occurred: {e}"
#     else:
#         result = "The chatbot is currently disabled."
#
#     return result
#
# def show_chatbot():
#     """Display the chatbot interface using Streamlit."""
#     st.title("Stock Market Chatbot")
#
#     # Input Area for User Prompt
#     user_input = st.text_area("Enter your stock-related question:", "", height=150)
#
#     if st.button("Send"):
#         if user_input.strip():
#             with st.spinner("Generating response..."):
#                 response = call_openai(user_input)
#             st.text_area("Chatbot's response:", response, height=150, disabled=True)
#         else:
#             st.warning("Please enter a message before sending.")
#
# # # Run the chatbot interface
# # if __name__ == "__main__":
# #     show_chatbot()
import os

# from openai import OpenAI
# import streamlit as st
# from dotenv import load_dotenv
#
# st.title("ChatGPT-like clone")
#
# load_dotenv()
#
# client = OpenAI()
#
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"
#
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# if prompt := st.chat_input("What is up?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
#
#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         )
#         response = st.write_stream(stream)
#     st.session_state.messages.append({"role": "assistant", "content": response})


# from openai import OpenAI
# import streamlit as st
# from dotenv import load_dotenv
#
# load_dotenv()
#
# client = OpenAI()
#
# st.title("Stock Market Expert Chatbot")
#
#
#
# # Set the default model
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"
#
# # Initialize messages list if it doesn't exist
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# # Define the system message to set the assistant's role as a stock market expert
# system_message = {
#     "role": "system",
#     "content": "You are a stock market expert. Provide insights, predictions, and explanations related to stock market trends and investments. If the query is unrelated to stocks or finance, respond that you are not an expert in that area."
# }
#
# # Display previous messages in the chat
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# # Handle user input
# if prompt := st.chat_input("Ask a question about stocks or finance:"):
#     # Append the user message to the session state
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
#
#     # Add the system message and user messages to the request
#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=[system_message] + [
#                 {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
#             ],
#             stream=True,
#         )
#         # Collect and display the assistant's response from the stream
#         response = st.write_stream(stream)
#
#     # Add the assistant's response to the session state
#     st.session_state.messages.append({"role": "assistant", "content": response})
#
