"""
ConditionDetector: Keyword-based detection of narrative condition triggers.

When an NPC agent's response contains keywords matching a condition's fingerprint,
and all prerequisite conditions (depends_on) are met, the condition is triggered.

Trigger fingerprints are loaded from triggers.json via ScriptManager (zero hardcoding).
"""


class ConditionDetector:
    def __init__(self, script_mgr):
        self.script_mgr = script_mgr

    def check(self, npc_name, response):
        """Check if agent response triggers any conditions.
        Returns list of newly triggered condition names.
        """
        triggered = []
        response_lower = response.lower()

        for cond in self.script_mgr.get_trigger_fingerprints():
            cond_name = cond["name"]

            # Skip if wrong NPC or already triggered
            if cond["trigger_npc"] != npc_name:
                continue
            if cond_name in self.script_mgr.conditions:
                continue

            # Check dependency
            dep = cond.get("depends_on")
            if dep and dep not in self.script_mgr.conditions:
                continue

            # Count keyword matches
            keywords = cond["keywords"]
            matches = sum(1 for kw in keywords if kw.lower() in response_lower)
            if matches >= cond["min_matches"]:
                self.script_mgr.mark_trigger_seen(npc_name, cond["trigger_fact_idx"])
                triggered.append(cond_name)
                print(f"[CONDITION] Triggered: '{cond_name}' (matched {matches}/{len(keywords)} keywords)")

        return triggered
