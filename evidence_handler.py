"""
EvidenceHandler: Handles evidence retrieval for the detective game.

Features:
- Loads evidence entries from ScriptManager.
- Two-stage matching: exact keyword match, then substring match.

Returns the corresponding evidence description or a fallback message.
"""
import re


class EvidenceHandler:
    def __init__(self, script_manager):
        self.script_mgr = script_manager
        self.evidence = self.script_mgr.raw_script.get("evidence", {})

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\u4e00-\u9fff ]', '', text)
        return text

    def search(self, item_name: str) -> str:
        query = self.normalize(item_name)

        # Stage 1: exact keyword match
        for key in self.evidence:
            if self.normalize(key) == query:
                return self.evidence[key]

        # Stage 1b: substring match
        for key in self.evidence:
            nk = self.normalize(key)
            if query in nk or nk in query:
                return self.evidence[key]

        return "未找到相关证据。"
