from typing import Dict, List, Optional


class ConversationMemory:

    def __init__(self):

        self.conversations: Dict[
            str,
            List[dict],
        ] = {}

        self.context: Dict[
            str,
            dict,
        ] = {}

    # --------------------------------------------------
    # Get conversation
    # --------------------------------------------------

    def get(
        self,
        conversation_id: str,
    ) -> List[dict]:

        return self.conversations.get(
            conversation_id,
            [],
        )

    # --------------------------------------------------
    # Add message
    # --------------------------------------------------

    def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ):

        if (
            conversation_id
            not in self.conversations
        ):

            self.conversations[
                conversation_id
            ] = []

        self.conversations[
            conversation_id
        ].append(
            {
                "role": role,
                "content": content,
            }
        )

    # --------------------------------------------------
    # Store entity context
    # --------------------------------------------------

    def set_context(
        self,
        conversation_id: str,
        entity_type: str,
        entity: str,
    ):

        self.context[
            conversation_id
        ] = {
            "entity_type": entity_type,
            "entity": entity,
        }

    # --------------------------------------------------
    # Get entity context
    # --------------------------------------------------

    def get_context(
        self,
        conversation_id: str,
    ) -> Optional[dict]:

        return self.context.get(
            conversation_id
        )

    # --------------------------------------------------
    # Clear conversation
    # --------------------------------------------------

    def clear(
        self,
        conversation_id: str,
    ):

        self.conversations.pop(
            conversation_id,
            None,
        )

        self.context.pop(
            conversation_id,
            None,
        )