import os, json, yaml, re
from typing import Any, Dict
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from state import State
from config import SERVICE_FOLDER, EMBEDDING_MODEL, CONVERTER_MODEL


class ConverterAgent:
    def __init__(self, llm_model: str = CONVERTER_MODEL, embedding_model: str = EMBEDDING_MODEL):
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.0,
            top_p=0.95,
            num_ctx=2048
        )
        self.api_spec_yaml: str | None = None
        self.system_message: str | None = None
        self.embedding = OllamaEmbeddings(model=embedding_model)

    def load_and_flatten_openapi(self, file_path: str) -> str:
        """Load YAML/JSON OpenAPI e return YAML 'flat'."""
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".json"):
                spec: Dict[str, Any] = json.load(f)
            else:
                spec: Dict[str, Any] = yaml.safe_load(f)

        base_url = spec.get("servers", [{}])[0].get("url", "")
        info = spec.get("info", {"title": "Unknown", "version": "1.0.0"})
        paths = spec.get("paths", {})

        flat_spec: Dict[str, Any] = {
            "openapi": "3.0.0",
            "info": {"title": info.get("title", "API"), "version": info.get("version", "1.0.0")},
            "servers": [{"url": base_url}],
            "paths": {},
        }

        for endpoint, methods in paths.items():
            for method, mdata in methods.items():
                schema = {}
                try:
                    schema_props = (
                        mdata.get("responses", {})
                        .get("200", {})
                        .get("content", {})
                        .get("application/json", {})
                        .get("schema", {})
                        .get("properties", {})
                    )
                    if schema_props:
                        schema = {k: v.get("type", "string") for k, v in schema_props.items()}
                except Exception:
                    pass

                flat_spec["paths"].setdefault(endpoint, {})
                flat_spec["paths"][endpoint][method] = {
                    "summary": mdata.get("summary", f"{method.upper()} {endpoint}"),
                    "parameters": mdata.get("parameters", []),
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "properties": schema}
                                }
                            },
                        }
                    },
                }

        return yaml.dump(flat_spec, sort_keys=False, allow_unicode=True)

    def html_to_openapi_with_llm(self, file_path: str) -> str:
        """Convert HTML API documentation into a validated OpenAPI YAML using LLM."""
        from openapi_spec_validator import validate_spec
        from openapi_spec_validator.validation.exceptions import OpenAPIValidationError

        output_dir = "generated_openapi"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, base_name + ".yaml")

        # Se già esiste, lo ricarichiamo
        if os.path.exists(output_file):
            print(f"Fetching the OpenAPI specifications for the file: {output_file}")
            return self.load_and_flatten_openapi(output_file)

        print("Generating OpenAPI specs from HTML documentation...")

        # Recupera documentazione indicizzata
        vectorstore = FAISS.load_local("faiss_index", self.embedding, allow_dangerous_deserialization=True)
        all_docs = vectorstore.docstore._dict.values()
        matching_docs = [
            doc.page_content for doc in all_docs
            if file_path in str(doc.metadata.get("source", ""))
        ]
        if not matching_docs:
            raise ValueError(f"No indexed content found for {file_path}")

        combined_content = "\n\n".join(matching_docs)

        # Prompt rinforzato
        prompt = f"""
You are an expert assistant capable of extracting information from an API's unstructured documentation and 
producing a specification in valid OpenAPI 3.0.0 YAML format.

- Return only the specification in valid YAML format, DO NOT INCLUDE comments, explanation, etc: the output MUST starts with "openapi: 3.".
- The returned specification must have all fields, including nested fields, in valid YAML format.
- Must include the fields: openapi, info, servers, and paths.

Documentation:
{combined_content[:12000]}
"""

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        openapi_yaml = response.content.strip()

        # Pulizia aggressiva
        openapi_yaml = re.sub(r"(?is)^.*?(openapi:\s*3\.\d+\.\d+)", r"\1", openapi_yaml)
        openapi_yaml = re.sub(r"```[a-zA-Z]*", "", openapi_yaml)
        openapi_yaml = re.sub(r"```", "", openapi_yaml).strip()
        openapi_yaml = re.split(r'\n\s*[A-Z][a-z]+\s', openapi_yaml)[0].strip()

        try:
            spec_dict = yaml.safe_load(openapi_yaml)
        except yaml.YAMLError as e:
            print("YAML parsing failed:", e)
            raise ValueError("Generated YAML is invalid. LLM output:\n" + openapi_yaml)
        """
        try:
            validate_spec(spec_dict)
        except OpenAPIValidationError as e:
            print("OpenAPI validation failed:", e)
            raise
        """
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write(openapi_yaml)

        print(f"OpenAPI spec validated and saved to {output_file}")
        return self.load_and_flatten_openapi(output_file)

    def load_api_spec(self, api_path: str) -> str:
        """Load API spec from file, converting HTML if needed."""
        if api_path.endswith(".html"):
            return self.html_to_openapi_with_llm(api_path)
        else:
            return self.load_and_flatten_openapi(api_path)

    def build_system_message(self) -> str:
        """Build the system message for the LLM using the loaded API spec."""
        self.system_message = f"""
        You have access to API tools that can send HTTP requests (GET/POST/PATCH/PUT/DELETE).
        When the user asks something, you must use the tools to actually send the request and return the real response data, even if it's an error code.
        Do NOT provide code samples or explain how to call the API.
        Use ONLY the endpoints/parameters documented below.
        Here is documentation on the API:
        {self.api_spec_yaml}
        """.strip()

        return self.system_message

    def run(self, state: State) -> State:
        print("Running ConverterAgent...")
        idx = state.get("current_index", 0)
        files = state.get("candidate_files", []) or []
        if idx >= len(files):
            return {**state, "done": True, "error": "No more files to process"}

        api_file = files[idx]
        api_path = os.path.join(SERVICE_FOLDER, os.path.basename(api_file))

        if not os.path.exists(api_path):
            return {**state, "current_index": idx + 1}

        try:
            self.api_spec_yaml = self.load_api_spec(api_path)
            system_message = self.build_system_message()

            return {
                **state,
                "current_api_path": api_path,
                "api_spec_yaml": self.api_spec_yaml,
                "system_message": system_message,
            }
        except Exception as e:
            print(f"Error processing {api_path}: {e}")
            return {**state, "current_index": idx + 1}
