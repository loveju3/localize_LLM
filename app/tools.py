"""A deliberately small allowlist of tools exposed to document chat."""
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_documents",
    "description": "搜尋使用者已上傳且在目前文件範圍內的段落。回答文件問題前使用。",
    "parameters": SearchArguments.model_json_schema(),
}}


class DocumentTools:
    def __init__(self, search, document_ids):
        self.search = search
        self.document_ids = document_ids
        self.sources = {}
        self.trace = []

    def execute(self, call):
        function = call.get("function", {})
        if function.get("name") != "search_documents":
            self.trace.append({"name": "unknown", "status": "rejected"})
            return {"error": "未知工具，僅可使用 search_documents"}
        try:
            arguments = SearchArguments.model_validate_json(function.get("arguments", ""))
        except (ValidationError, ValueError, TypeError):
            self.trace.append({"name": "search_documents", "status": "invalid_arguments"})
            return {"error": "參數必須只有 question：1 至 2000 字的非空白字串"}
        rows = self.search(arguments.question, self.document_ids)
        results = []
        for row in rows:
            key = row["chunk_id"]
            if key not in self.sources:
                self.sources[key] = {**row, "source_id": f"S{len(self.sources) + 1}"}
            results.append(self.sources[key])
        self.trace.append({"name": "search_documents", "question": arguments.question,
                           "status": "ok", "source_ids": [r["source_id"] for r in results]})
        return {"sources": results}


def tool_message(call, result):
    return {"role": "tool", "tool_call_id": call["id"],
            "content": json.dumps(result, ensure_ascii=False)}
