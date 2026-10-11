from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage


def get_message_for_llm(
    model_id: str,
    system_prompt: str,
    context: str,
    question: str,
    demonstrations_block: str = "",
) -> List[Dict[str, str]]:
    user_content = (
        f"{demonstrations_block}"
        f"<context>{context}</context>\n<question>{question}</question>"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    return messages
