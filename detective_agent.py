"""
DetectiveAgent: An AI detective that autonomously investigates the murder case.

Uses DeepSeek (OpenAI-compatible) tool use to drive a two-phase agentic loop:

  Phase 1 — Motive + Evidence (parallel)
    ask_npc / search_evidence collect clues.
    When all motive clues are found, the tool result hints to shift focus to evidence.
    When all evidence clues are found, deduce_crime_process is unlocked.

  Phase 2 — Deduction
    deduce_crime_process: submit the physical crime mechanism (required before accusation).
    make_accusation: final accusation, blocked until both phases complete.
"""

from llm_router import LLMRouter

try:
    from langsmith import trace as ls_trace
except ImportError:
    class ls_trace:  # no-op context manager when langsmith not installed
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass

EVIDENCE_CONDITIONS = [
    "Daniel的眼镜腿是弯的",
    "Morgan平时戴眼镜",
    "Morgan今天没戴眼镜",
]

MOTIVE_CONDITIONS = [
    "Morgan突然签了合同",
    "Morgan和Daniel有经济纠纷",
]

REQUIRED_CONDITIONS = EVIDENCE_CONDITIONS + MOTIVE_CONDITIONS

def condition_hint(condition: str) -> str:
    if condition == "Daniel的眼镜腿是弯的":
        return "现场眼镜完好无损，但Jordan提过Daniel的眼镜腿弯了 → 为什么现场遗留的眼镜是完好的？难道被掉包了？"
    if condition == "Morgan平时戴眼镜":
        return "Lucas说Morgan平时会戴眼镜，但案发当天没戴 → 为什么刻意不戴？还是说发生了什么让他的眼镜无法戴上？"
    if condition == "Morgan今天没戴眼镜":
        return "Morgan声称眼镜忘在家里 → 结合平时必须戴眼镜，看来有重大嫌疑。"
    if condition == "Morgan突然签了合同":
        return "Morgan在压力下签了搬迁补贴协议 → 背后存在经济动机"
    if condition == "Morgan和Daniel有经济纠纷":
        return "Daniel以贷款陷阱骗Morgan签合同 → 直接的杀人动机"
    return ""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_npc",
            "description": (
                "向一位嫌疑人提问并获取其回答。"
                "每次提问后系统会自动检测是否触发新线索，并更新嫌疑人的已知信息。"
                "同一问题不要重复提问。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "npc_name": {
                        "type": "string",
                        "enum": ["Jordan", "Lucas", "Morgan"],
                        "description": "要审讯的嫌疑人姓名",
                    },
                    "question": {
                        "type": "string",
                        "description": "向嫌疑人提出的问题（中文口语，20字以内）",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "本轮决策的推理依据：基于已知线索，为什么选这个人问这个问题（1-2句话）",
                    },
                },
                "required": ["npc_name", "question", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": "查询现场证物的详细检验报告。可用证物：炸鸡、眼镜、水杯、餐具、餐巾纸。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "证物名称",
                    },
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deduce_crime_process",
            "description": (
                "提交对作案过程的推断：毒素施加在哪个物体上，以及物理传递机制。"
                "需要在找到全部 3 条物证线索后才能提交，否则会被驳回。"
                "提交成功后才能进行最终指控。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "enum": ["水杯", "餐具", "餐巾纸", "炸鸡", "眼镜"],
                        "description": "毒素被施加在哪个物体上（从选项中选择）",
                    },
                    "process": {
                        "type": "string",
                        "description": "作案过程的完整描述：凶手何时何地对该物体施毒，毒素如何最终进入受害者体内（2-3句话）",
                    },
                },
                "required": ["location", "process"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_accusation",
            "description": (
                "提交最终指控（凶手 + 动机）。"
                "必须先成功提交 deduce_crime_process，且触发全部 5 条线索，否则会被驳回。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "culprit": {
                        "type": "string",
                        "enum": ["Jordan", "Morgan", "Lucas"],
                        "description": "凶手姓名（从选项中选择）",
                    },
                    "motive": {
                        "type": "string",
                        "enum": ["合同经济纠纷", "职场羞辱", "被迫搬迁店铺"],
                        "description": "作案动机（从选项中选择）",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "完整推理过程（2-4句话）",
                    },
                },
                "required": ["culprit", "motive", "reasoning"],
            },
        },
    },
]

DETECTIVE_SYSTEM_PROMPT = """你是一名经验丰富的侦探，正在调查一起案件。

案件背景：
Daniel，一家房地产公司的中层经理，在 Lucas 经营的小餐馆被发现因急性氰化钾中毒死亡，死亡时间约11:45。
三位嫌疑人：
- Jordan：Daniel 的下属，紧张、安静的初级员工
- Lucas：餐馆老板，脾气火爆，对 Daniel 的行事方式极为不满
- Morgan：隔壁理发店老板，Daniel 当天上午曾去其店

现场遗留证物：炸鸡、眼镜、水杯、餐具、餐巾纸

调查原则：
1. search_evidence 只返回法医的客观检验数据；真正的线索来自嫌疑人对证物的主观陈述——他们亲眼看到了什么、注意到了什么异常。必须主动询问 NPC 关于现场证物的观察。
2. 发现新线索后，应重新审讯相关嫌疑人，因为新情况可能让他们说出之前隐瞒的信息。
3. 毒素不一定直接施加在受害者身上，可能通过间接接触链传递。
4. 调查分两个阶段：先收集全部线索，再依次提交 deduce_crime_process 和 make_accusation。
5. 如果5轮对话还没找到新线索，尝试用目前现有的线索询问不同npc。
6. 证物线索找齐后可以尝试转为询问动机线索。动机线索找齐后可以尝试专注询问证物线索。"""


class DetectiveAgent:
    def __init__(self, agent_manager, script_mgr, evidence_handler, router: LLMRouter,
                 judge_agent=None):
        self.agent_manager        = agent_manager
        self.script_mgr           = script_mgr
        self.evidence_handler     = evidence_handler
        self.router               = router
        self.judge_agent          = judge_agent
        self.max_turns             = 25
        self.crime_process_deduced = False
        self.deduced_location      = None
        self._accusation_accepted  = False
        self._investigation_log    = []   # {"npc", "question", "answer"}
        self._final_accusation     = {}   # filled when make_accusation succeeds
        self._deduced_process      = ""   # stored when deduce_crime_process succeeds
        self._last_judge_verdict   = None  # {"approved": bool, "feedback": str}

    def run_stream(self):
        """Agentic tool-use loop. Yields SSE event dicts for the frontend."""
        yield {"type": "start", "message": "AI侦探开始调查..."}

        messages = [{"role": "user", "content": self._build_initial_prompt()}]

        with ls_trace("detective_investigation", run_type="chain"):
            for turn in range(self.max_turns):
                yield {"type": "thinking", "turn": turn + 1}

                response = self.router.chat(
                    messages=messages,
                    tools=TOOLS,
                    system=DETECTIVE_SYSTEM_PROMPT,
                    max_tokens=1024,
                    temperature=0.2,
                )

                messages.append(self.router.make_assistant_message(response))

                if response.finish_reason == "stop":
                    if (self.crime_process_deduced
                            and self.script_mgr.has_solved_all_conditions()
                            and not self._accusation_accepted):
                        messages.append({"role": "user", "content": "作案过程已确认，所有线索已收集，请立即调用 make_accusation 提交最终指控。"})
                        continue
                    break

                if response.finish_reason != "tool_calls":
                    break

                should_stop = False
                prev_conditions = set(self.script_mgr.get_required_conditions())

                for tc in response.tool_calls:
                    result_content, stop = self._execute_tool(tc.name, tc.input)
                    messages.append(self.router.make_tool_result_message(tc.id, result_content))
                    if stop:
                        should_stop = True

                    for event in self._tool_events(tc.name, tc.input, turn + 1):
                        yield event

                # Per-turn conditions summary appended as a user message
                conditions = self.script_mgr.get_required_conditions()
                if conditions:
                    self.max_turns += 5  # extend max turns if any conditions found to allow for follow-up questioning
                evidence_done = all(c in conditions for c in EVIDENCE_CONDITIONS)
                motive_done   = all(c in conditions for c in MOTIVE_CONDITIONS)
                summary = (
                    f"物证线索 ({sum(c in conditions for c in EVIDENCE_CONDITIONS)}/{len(EVIDENCE_CONDITIONS)}): "
                    f"{', '.join(c for c in EVIDENCE_CONDITIONS if c in conditions) or '暂无'}\n"
                    f"动机线索 ({sum(c in conditions for c in MOTIVE_CONDITIONS)}/{len(MOTIVE_CONDITIONS)}): "
                    f"{', '.join(c for c in MOTIVE_CONDITIONS if c in conditions) or '暂无'}"
                )
                new_conditions = set(conditions) - prev_conditions
                for c in new_conditions:
                    hint = condition_hint(c)
                    if hint:
                        summary += f"\n\n🔍 新线索「{c}」：{hint}"

                if motive_done and not evidence_done:
                    summary += "\n\n💡 动机已查明，建议专注物证：询问嫌疑人关于现场证物的观察。"
                if evidence_done and not self.crime_process_deduced:
                    summary += "\n\n💡 物证线索已齐全，请调用 deduce_crime_process 提交作案过程推断。"
                if self.crime_process_deduced and not motive_done:
                    summary += "\n\n💡 作案过程已查明，建议转向动机：询问嫌疑人 Morgan 与死者的矛盾。"
                messages.append({"role": "user", "content": summary})

                if should_stop:
                    break

        yield {"type": "done"}

    # ── Tool execution ──────────────────────────────────────────────────────

    def _execute_tool(self, name, inp):
        """Execute a tool call. Returns (result_str, should_stop)."""

        if name == "ask_npc":
            npc      = inp["npc_name"]
            question = inp["question"]
            answer   = self.agent_manager.ask(npc, question)
            return f"[{npc}的回答]: {answer}", False

        if name == "search_evidence":
            return self.evidence_handler.search(inp["item"]), False

        if name == "deduce_crime_process":
            conditions    = self.script_mgr.get_required_conditions()
            evidence_done = all(c in conditions for c in EVIDENCE_CONDITIONS)

            if not evidence_done:
                missing = [c for c in EVIDENCE_CONDITIONS if c not in conditions]
                missing_text = "\n".join(
                    f"  - 「{c}」（{condition_hint(c)}）" for c in missing
                )
                return (
                    f"❌ 推断被驳回：物证线索不足，还缺：\n{missing_text}\n请继续调查物证。"
                ), False

            self.deduced_location = inp.get("location", "")
            self._deduced_process = inp.get("process", "")

            # Judge evaluates the mechanism before accepting
            if self.judge_agent:
                verdict = self.judge_agent.evaluate(
                    self._investigation_log,
                    {"location": self.deduced_location, "process": self._deduced_process},
                )
                self._last_judge_verdict = verdict
                if not verdict["approved"]:
                    # Reset so detective must try again
                    self.deduced_location = ""
                    self._deduced_process = ""
                    return (
                        f"❌ 作案过程推断被评审驳回。\n"
                        f"评审意见：{verdict['feedback']}\n"
                        f"请重新分析证据，修正推断后再次提交。"
                    ), False
            else:
                self._last_judge_verdict = None

            self.crime_process_deduced = True
            return (
                f"✅ 作案过程推断已通过评审。"
                + (f"\n评审意见：{self._last_judge_verdict['feedback']}" if self._last_judge_verdict else "")
                + "\n现在可以提交最终指控（make_accusation）。"
            ), False

        if name == "make_accusation":
            conditions = self.script_mgr.get_required_conditions()
            all_met    = self.script_mgr.has_solved_all_conditions()

            if not self.crime_process_deduced:
                return "❌ 指控被驳回：尚未提交作案过程推断，请先调用 deduce_crime_process。", False

            if not all_met:
                missing = [
                    f"「{c}」（{condition_hint(c)}）"
                    for c in REQUIRED_CONDITIONS if c not in conditions
                ]
                missing_text = "\n".join(f"  - {m}" for m in missing)
                return (
                    f"❌ 指控被驳回：还有 {len(missing)} 条线索未发现：\n{missing_text}\n请继续调查。"
                ), False

            self._accusation_accepted = True
            return "✅ 指控已受理。", True

        return f"[未知工具: {name}]", False

    def _tool_events(self, name, inp, turn):
        """Yield SSE events corresponding to a tool call."""

        if name == "ask_npc":
            npc       = inp["npc_name"]
            question  = inp["question"]
            reasoning = inp.get("reasoning", "")
            agent     = self.agent_manager.agents.get(npc)
            answer    = agent.conversation_history[-1]["content"] if (agent and agent.conversation_history) else ""

            # Accumulate for judge
            self._investigation_log.append({"npc": npc, "question": question, "answer": answer})

            if reasoning:
                yield {"type": "thought", "turn": turn, "reasoning": reasoning, "target": npc, "question": question}
            yield {"type": "question", "turn": turn, "npc": npc, "text": question}
            yield {"type": "answer",   "turn": turn, "npc": npc, "text": answer}

            conditions = self.script_mgr.get_required_conditions()
            yield {"type": "observe", "conditions": conditions, "all_met": self.script_mgr.has_solved_all_conditions()}

        elif name == "search_evidence":
            yield {"type": "evidence", "item": inp["item"], "result": self.evidence_handler.search(inp["item"])}

        elif name == "deduce_crime_process":
            verdict = getattr(self, "_last_judge_verdict", None)
            yield {
                "type":     "deduction",
                "location": inp.get("location", ""),
                "process":  inp.get("process", ""),
                "approved": verdict["approved"] if verdict else True,
                "feedback": verdict["feedback"] if verdict else "",
            }

        elif name == "make_accusation":
            culprit  = inp.get("culprit", "")
            motive   = inp.get("motive", "")
            location = self.deduced_location or ""
            correct  = (
                self._accusation_accepted
                and culprit  == "Morgan"
                and location == "眼镜"
                and motive   == "合同经济纠纷"
            )
            yield {
                "type":      "accusation",
                "accepted":  self._accusation_accepted,
                "culprit":   culprit,
                "location":  location,
                "motive":    motive,
                "reasoning": inp.get("reasoning", ""),
                "correct":   correct,
            }
            self._accusation_accepted = False  # reset for potential retries

    # ── Prompt building ─────────────────────────────────────────────────────

    def _build_initial_prompt(self):
        evidence      = self.script_mgr.get_evidence()
        evidence_text = "\n".join(f"  【{k}】{v}" for k, v in evidence.items())

        return f"""法医报告摘要：
- 死者右手食指与中指指腹检出氰化钾残留，提示毒素来自接触某个物体表面而非直接涂抹

现场证物：
{evidence_text}

请开始调查。建议如下：1.询问被害者和嫌疑人们当天的时间线。2.询问每位嫌疑人对现场证物的观察，再结合法医报告寻找矛盾点。3.询问嫌疑人们和被害者的关系以及嫌疑人之间的关系。
调查分两阶段：①收集全部线索 → ②提交 deduce_crime_process，再提交 make_accusation。"""
