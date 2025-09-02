from pipeline import run_with_multiagent
from config import SERVICE_FOLDER

if __name__ == "__main__":
    query = input("Ask me a question. I'll respond using the services described in the '" + SERVICE_FOLDER + "' folder.\n>> ")
    query = query + ("\nDon't explain how to call the API and don't show code examples. "
        "You must use your tool to actually send the request and return the real response data.\nReturn ONLY the http response."
    )
    run_with_multiagent(query)