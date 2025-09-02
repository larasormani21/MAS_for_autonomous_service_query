import os
import json, yaml
from typing import List
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataIngestor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_html(self, file_path: str) -> List[Document]:
        """
        Loads an HTML file and transforms it into a list of Documents.
        Uses BeautifulSoup to parse and clean the HTML, then splits into sections.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        for tag in soup.select(
            "nav, footer, header, noscript, aside, style, "
            ".sidebar, .menu, .navigation, .footer, .header, "
            ".ad, .ads, .advertisement"
        ):
            tag.extract()

        docs = []
        current_section = ""
        current_text = ""

        for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
            if tag.name.startswith("h"):
                if current_text.strip():
                    docs.append(Document(
                        page_content=current_text.strip(),
                        metadata={"section": current_section, "source": file_path}
                    ))
                current_section = tag.get_text(separator=" ", strip=True)
                current_text = f"# {current_section}\n"
            else:
                text = tag.get_text(separator=" ", strip=True)
                if text:
                    current_text += text + "\n"

        if current_text.strip():
            docs.append(Document(
                page_content=current_text.strip(),
                metadata={"section": current_section, "source": file_path}
            ))

        return self.chunk_docs(docs)

    def load_openapi_yaml(self, file_path: str) -> List[Document]:
        """
        Loads an OpenAPI YAML file and transforms it into a list of Documents.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            yaml_obj = yaml.safe_load(f)

        return self._parse_openapi_dict(yaml_obj, file_path)

    def load_openapi_json(self, file_path: str) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)
        return self._parse_openapi_dict(json_obj, file_path)

    def _parse_openapi_dict(self, spec: dict, file_path: str) -> List[Document]:
        docs = []
        info = spec.get("info", {})
        base_metadata = {
            "api_title": info.get("title", ""),
            "api_version": info.get("version", ""),
            "source": file_path
        }

        if info:
            content = f"API Info:\n" + json.dumps(info, indent=2)
            docs.append(Document(page_content=content, metadata={**base_metadata, "section": "info"}))
        servers = spec.get("servers", [])
        if servers:
            content = f"Servers:\n" + json.dumps(servers, indent=2)
            docs.append(Document(page_content=content, metadata={**base_metadata, "section": "servers"}))
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, op in methods.items():
                summary = op.get("summary", "")
                description = op.get("description", "")
                parameters = op.get("parameters", [])
                request_body = op.get("requestBody", {})
                responses = op.get("responses", {})

                param_texts = []
                for p in parameters:
                    pname = p.get("name", "")
                    pdesc = p.get("description", "")
                    param_texts.append(f"- {pname}: {pdesc}")
                param_block = "\n".join(param_texts) if param_texts else "None"

                request_block = ""
                if request_body:
                    content_obj = request_body.get("content", {})
                    for content_type, schema in content_obj.items():
                        request_block += f"\n{content_type}: {json.dumps(schema.get('schema', {}), indent=2)}"
                request_block = request_block if request_block else "None"

                response_block = ""
                for code, resp in responses.items():
                    rdesc = resp.get("description", "")
                    response_block += f"\n{code}: {rdesc}"
                response_block = response_block if response_block else "None"

                content = f"""### {method.upper()} {path}
Summary: {summary}
Description: {description}
Parameters:
{param_block}
Request Body:
{request_block}
Responses:
{response_block}
"""
                docs.append(Document(
                    page_content=content.strip(),
                    metadata={**base_metadata, "method": method, "path": path}
                ))

        components = spec.get("components", {})
        if components:
            content = "Components:\n" + json.dumps(components, indent=2)
            docs.append(Document(page_content=content, metadata={**base_metadata, "section": "components"}))

        return self.chunk_docs(docs)
    
    def chunk_docs(self, docs: List[Document]) -> List[Document]:
        """
        Applies recursive chunking to a list of Documents.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = splitter.split_documents(docs)
        return self.remove_duplicates(split_docs)

    def remove_duplicates(self, docs: List[Document]) -> List[Document]:
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs
