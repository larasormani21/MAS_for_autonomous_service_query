from langchain_ollama import OllamaLLM
from state import State
from config import EXTRACTOR_MODEL  

EXTRACT_PROMPT = """
You are an agent that converts raw API responses into a concise, human-readable answer.

Rules:
1. Read the raw API response (JSON, XML, or plain text).
2. Extract the information relevant to the user's original question.
3. Reply ONLY with a short, clear textual answer. Do NOT include code or JSON.
4. Use ONLY the API response to answer, NOT other information.

If the API response is already a valid and well-constructed answer to the User question, you can return the API response as is.

User question: {query}
API response: {response}
"""

class ExtractorAgent:
    """
    ExtractorAgent takes the raw API response and produces a concise, human-readable answer.
    It uses an LLM to reformat or extract only the relevant information.
    """

    def __init__(self, llm_model: str = EXTRACTOR_MODEL):
        self.llm = OllamaLLM(model=llm_model)

    def run(self, state: State) -> State:
        print("Running ExtractorAgent...")
        last_response = state.get("last_response")
        user_query = state.get("user_query", "")

        if not last_response:
            return state

        prompt = EXTRACT_PROMPT.format(query=user_query, response=last_response)

        try:
            print("Extracting human-readable answer from API response...")
            extracted = self.llm.invoke(prompt)
            print(f"Extracted answer:\n{extracted}\n")
            return {
                **state,
                "last_response": extracted,
                "done": True
            }
        except Exception as e:
            return {
                **state,
                "error": f"ExtractorAgent error: {e}"
            }
