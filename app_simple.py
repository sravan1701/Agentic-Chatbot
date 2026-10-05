import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

models = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

prompt = "Say hello in one short sentence."

for model_name in models:

    print("\n" + "=" * 50)
    print("MODEL:", model_name)
    print("=" * 50)

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name
        )

        start = time.time()
        first_chunk_time = None
        full_response = ""

        for chunk in llm.stream(prompt):

            if first_chunk_time is None:
                first_chunk_time = time.time() - start

            content = chunk.content

            if isinstance(content, str):
                full_response += content

            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        full_response += block.get("text", "")

        total_time = time.time() - start

        print("Time to first chunk:", round(first_chunk_time, 2), "seconds")
        print("Total time:", round(total_time, 2), "seconds")
        print("Response:", full_response)

    except Exception as e:
        print("ERROR:", e)