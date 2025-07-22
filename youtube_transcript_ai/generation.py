from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, convert_system_message_to_human=True)

def generate_output(final_prompt):
    answer = llm.invoke(final_prompt)
    print(answer.content)
    return answer