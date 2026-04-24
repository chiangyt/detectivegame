"""
ScriptManager.py

Manages structured script data and dynamic updates in the detective game.

Core responsibilities:
- Loads case data (case.json), NPC facts (*.json), and trigger rules (triggers.json) from scripts/ folder.
- Tracks triggered conditions and manages stateful replacements of NPC facts.
- Applies conditional overrides when specific conditions are met, enabling branching narrative behavior.
- Provides helper functions to access raw data, track condition triggers, and evaluate deduction readiness.
"""
import json
import os
import glob as _glob


class ScriptManager:
    def __init__(self, path):
        """Load game data from scripts/ directory next to *path*.

        Layout expected::

            scripts/
                case.json         case_description + evidence
                triggers.json     conditions, keyword fingerprints, overrides
                jordan.json       {npc_name, facts[]}
                morgan.json
                lucas.json
        """
        base_dir = os.path.dirname(path) or "."
        scripts_dir = os.path.join(base_dir, "scripts")

        # --- load case data ---
        case_path = os.path.join(scripts_dir, "case.json")
        with open(case_path, encoding="utf-8") as f:
            self.raw_script = json.load(f)

        # --- load NPC facts ---
        self.raw_script["npcs"] = []
        for npc_file in sorted(_glob.glob(os.path.join(scripts_dir, "*.json"))):
            basename = os.path.basename(npc_file)
            if basename in ("case.json", "triggers.json"):
                continue
            with open(npc_file, encoding="utf-8") as f:
                npc_data = json.load(f)
                # store a copy of original facts for override reset
                npc_data["base_facts"] = list(npc_data["facts"])
                self.raw_script["npcs"].append(npc_data)

        # --- load triggers ---
        triggers_path = os.path.join(scripts_dir, "triggers.json")
        with open(triggers_path, encoding="utf-8") as f:
            self.triggers_data = json.load(f)

        # --- state ---
        self.conditions = set()
        self.override_applied_pos = set()   # (npc_name, fact_idx) pairs
        self.confirmed_trigger_hits = set() # (npc_name, fact_idx) pairs

        # --- build lookup maps from triggers.json ---
        # trigger_map: (npc_name, fact_idx) -> condition_name
        self.trigger_map = {}
        # condition_depends_on: condition_name -> prerequisite condition_name
        self.condition_depends_on = {}

        for cond in self.triggers_data.get("conditions", []):
            name = cond["name"]
            key = (cond["trigger_npc"], cond["trigger_fact_idx"])
            self.trigger_map[key] = name
            if cond.get("depends_on"):
                self.condition_depends_on[name] = cond["depends_on"]

        self.apply_overrides()

    # --- accessors ---

    def get_evidence(self):
        return self.raw_script.get("evidence", {})

    def get_case_description(self):
        return self.raw_script.get("case_description", "")

    def get_npcs(self):
        return self.raw_script.get("npcs", [])

    def get_conditions(self):
        return list(self.conditions)

    def get_required_conditions(self):
        """Return only triggered conditions that are required for deduction (excludes distractors)."""
        required = self.triggers_data.get("required_for_deduction", [])
        return [c for c in required if c in self.conditions]

    def get_required_condition_names(self):
        """Return all condition names required for deduction, in display order."""
        return list(self.triggers_data.get("required_for_deduction", []))

    def get_trigger_fingerprints(self):
        """Return conditions list from triggers.json (used by ConditionDetector)."""
        return self.triggers_data.get("conditions", [])

    # --- condition management ---

    def add_condition(self, flag):
        if flag not in self.conditions:
            self.conditions.add(flag)
            self.apply_overrides()

    def mark_trigger_seen(self, npc_name, fact_idx):
        """Record that a player has encountered a specific NPC fact line.

        If this line is a trigger for a condition and all dependencies are met,
        the condition is activated.
        """
        key = (npc_name, fact_idx)
        self.confirmed_trigger_hits.add(key)

        cond = self.trigger_map.get(key)
        if not cond:
            return

        dep = self.condition_depends_on.get(cond)
        if dep:
            if dep not in self.conditions:
                return
            if key not in self.override_applied_pos:
                return

        self.add_condition(cond)

    def apply_overrides(self):
        """Apply all condition-based fact overrides.

        1. Reset all NPC facts to base_facts.
        2. For each fulfilled condition, replace target facts.
        """
        # Reset
        for npc in self.get_npcs():
            npc["facts"] = list(npc["base_facts"])

        # Build name -> npc lookup
        npc_by_name = {n["npc_name"]: n for n in self.get_npcs()}

        # Apply
        for cond in self.triggers_data.get("conditions", []):
            if cond["name"] not in self.conditions:
                continue
            for ov in cond.get("overrides", []):
                target_npc = npc_by_name.get(ov["npc_name"])
                if not target_npc:
                    continue
                idx = ov["fact_idx"]
                if idx < len(target_npc["facts"]):
                    target_npc["facts"][idx] = ov["replace_text"]
                    self.override_applied_pos.add((ov["npc_name"], idx))

    def has_solved_all_conditions(self):
        required = set(self.triggers_data.get("required_for_deduction", []))
        return required.issubset(self.conditions)
