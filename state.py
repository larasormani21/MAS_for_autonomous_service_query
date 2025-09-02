from typing import TypedDict, List, Optional

class State(TypedDict, total=False):
    user_query: str
    candidate_files: List[str]
    current_index: int
    current_api_path: Optional[str]
    api_spec_yaml: Optional[str]
    system_message: Optional[str]
    last_response: Optional[str]
    done: bool
    retrieved: bool
    error: Optional[str]
    fetched_url: bool
    accepted: bool
    needs_reindex: bool
