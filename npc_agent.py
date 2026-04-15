"""
NPCAgent: A single NPC character powered by an LLM with structured knowledge.

Each agent has:
- A fixed personality
- A numbered knowledge base (facts list, updated when conditions trigger overrides)
- Conversation history (last 10 exchanges)
- Strict rules against fabrication
"""


class NPCAgent:
    def __init__(self, npc_name, personality, facts, openai_client, model="gpt-4o-mini"):
        self.npc_name = npc_name
        self.personality = personality
        self.openai_client = openai_client
        self.model = model
        self.conversation_history = []
        self.system_prompt = self._build_system_prompt(facts)

    def _build_system_prompt(self, facts):
        """Build system prompt from personality + current knowledge facts."""
        facts_block = ""
        for i, fact in enumerate(facts, 1):
            facts_block += f"{i}. {fact}\n"

        return (
            f"你是{self.npc_name}，一起侦探调查中的角色。请始终用中文回答。\n\n"
            f"## 你的身份\n{self.personality}\n\n"
            f"## 案件背景\n"
            f"Daniel，一家房地产公司的中层经理，在一家小餐馆被发现因急性氰化钾中毒死亡，"
            f"死亡时间约为上午11:45。你正在接受侦探的问询。\n\n"
            f"## 你所知道的（这是你唯一的事实）\n"
            f"你只知道以下事实。你绝对不能编造、捏造或暗示以下未列出的任何信息。"
            f"如果被问到你不知道的事情，就说你不知道或者没注意到。\n"
            f"{facts_block}\n"
            f"## 规则：\n"
            f"1. 只能引用上述事实列表中的内容。绝不编造新的事实。\n"
            f"2. 可以自然地复述事实，但必须保留所有事实性内容。\n"
            f"3. 如果被问到知识范围外的事情，就说不知道。\n"
            f"4. 保持角色扮演。对话要自然，不要像机器人。\n"
            f"5. 不要主动透露信息——只回答被问到的。但当你提到某条事实时，必须完整陈述该条事实的全部内容，不能只说一半就停下。每条事实是一个不可分割的整体。\n"
            f"6. 每次回复控制在2-4句话，除非问题需要更多。\n"
            f"7. 绝对不要暴露你是AI或者你有知识库。\n"
            f"8. 始终用中文回答，保持口语化。\n"
        )

    def rebuild_system_prompt(self, facts):
        """Rebuild system prompt with updated knowledge after condition overrides."""
        self.system_prompt = self._build_system_prompt(facts)

    def ask(self, question):
        """Send question to LLM agent and return response. Maintains conversation history."""
        if not self.openai_client:
            return "[No OpenAI API key set.]"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": question},
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=150,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            return f"[GPT Error] {str(e)}"

        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": reply})

        # Keep last 10 exchanges (20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return reply

    def ask_stream(self, question):
        """Stream response from LLM. Yields text chunks. Saves full reply to history when exhausted."""
        if not self.openai_client:
            self._save_to_history(question, "[No OpenAI API key set.]")
            yield "[No OpenAI API key set.]"
            return

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": question},
        ]

        full_reply = []
        try:
            stream = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=150,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_reply.append(delta)
                    yield delta
        except Exception as e:
            err = f"[GPT Error] {str(e)}"
            full_reply.append(err)
            yield err

        self._save_to_history(question, "".join(full_reply))

    def _save_to_history(self, question, reply):
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": reply})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
