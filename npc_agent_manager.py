"""
NPCAgentManager: Orchestrates 3 NPC agents for the detective game.

Responsibilities:
- Creates one NPCAgent per NPC at startup
- Routes player questions to the correct agent
- After each response, checks for condition triggers via ConditionDetector
- When overrides mutate dialogue, refreshes affected agents' system prompts
"""

from npc_agent import NPCAgent
from condition_detector import ConditionDetector

NPC_PERSONALITIES = {
    "Morgan": (
        "你是一个疲惫而谨慎的理发店老板。你说话小心，措辞考量。"
        "你不喜欢Daniel，但很少公开表露。你在这个街区住了很长时间。"
    ),
    "Lucas": (
        "你是一个脾气暴躁的餐馆老板。你说话直来直去，"
        "别人提到Daniel的时候你会明显不耐烦。你为自己的厨艺感到骄傲，"
        "很反感Daniel的恐吓手段。"
    ),
    "Jordan": (
        "你是房地产公司一个紧张、安静的初级员工。你说话犹犹豫豫，"
        "经常用语气词，避免说任何可能给自己惹麻烦的话。"
        "你是Daniel的下属，你怕他。"
    ),
}


class NPCAgentManager:
    def __init__(self, script_mgr, openai_client, model="gpt-4o-mini"):
        self.script_mgr = script_mgr
        self.openai_client = openai_client
        self.model = model
        self.detector = ConditionDetector(script_mgr)

        # Snapshot conditions before building agents
        self._prev_conditions = set(script_mgr.conditions)

        self.agents = {}
        for npc in script_mgr.get_npcs():
            name = npc["npc_name"]
            personality = NPC_PERSONALITIES.get(name, "a character in a detective game")
            self.agents[name] = NPCAgent(
                npc_name=name,
                personality=personality,
                facts=npc["facts"],
                openai_client=openai_client,
                model=model,
            )
        print(f"[AGENT] Initialized {len(self.agents)} NPC agents: {list(self.agents.keys())}")

    def ask(self, npc_name, question):
        """Route question to the correct NPC agent, then check for condition triggers."""
        agent = self.agents.get(npc_name)
        if not agent:
            return f"[Error] Unknown NPC: {npc_name}"

        response = agent.ask(question)

        # Check if the response triggers any conditions
        triggered = self.detector.check(npc_name, response)

        # If new conditions were added, overrides may have changed other NPCs' dialogue.
        # Refresh those agents' system prompts.
        if triggered:
            self._refresh_affected_agents()

        return response

    def ask_stream(self, npc_name, question):
        """Stream NPC response. Yields {"chunk": text} dicts, then a final {"done": True, ...} dict."""
        agent = self.agents.get(npc_name)
        if not agent:
            yield {"error": f"Unknown NPC: {npc_name}"}
            return

        full_reply = []
        for chunk in agent.ask_stream(question):
            full_reply.append(chunk)
            yield {"chunk": chunk}

        # Condition detection runs after the full response is collected
        triggered = self.detector.check(npc_name, "".join(full_reply))
        if triggered:
            self._refresh_affected_agents()

        yield {
            "done": True,
            "conditions": self.script_mgr.get_conditions(),
            "all_conditions_met": self.script_mgr.has_solved_all_conditions(),
        }

    def _refresh_affected_agents(self):
        """Rebuild system prompts for all agents whose knowledge may have changed."""
        current_conditions = set(self.script_mgr.conditions)
        if current_conditions == self._prev_conditions:
            return

        new_conds = current_conditions - self._prev_conditions
        self._prev_conditions = current_conditions

        # Refresh ALL agents since overrides can affect any NPC
        for npc in self.script_mgr.get_npcs():
            name = npc["npc_name"]
            if name in self.agents:
                self.agents[name].rebuild_system_prompt(npc["facts"])

        print(f"[AGENT] Refreshed agent knowledge after new conditions: {new_conds}")
