from typing import Dict, List


class ConversationMemory:

    def __init__(self):
        self.conversations: Dict[str, List[dict]] = {}

    def get(self, conversation_id: str) -> List[dict]:
        return self.conversations.get(
            conversation_id,
            []
        )

    def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append(
            {
                "role": role,
                "content": content,
            }
        )

    def clear(self, conversation_id: str):
        self.conversations.pop(
            conversation_id,
            None,
        )