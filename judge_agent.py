"""
JudgeAgent: Uses Claude to evaluate the detective's crime-process deduction.

Claude receives:
  - Case background and forensic evidence (public facts only)
  - All NPC dialogue collected during the investigation
  - The detective's proposed crime mechanism (location + process)

Claude does NOT receive the correct answer.

evaluate() returns {"approved": bool, "feedback": str} via tool use.
The caller (DetectiveAgent._execute_tool) uses this to accept or reject
the deduction and feed the verdict back to DeepSeek as a tool result.
"""

from llm_router import LLMRouter

JUDGE_SYSTEM_PROMPT = """你是一名严格的法庭评审员，负责审核侦探提交的作案过程推断。

你的审核标准：
1. 作案机制能否解释法医报告中氰化钾的实际分布（受害者手指、炸鸡盘边缘、水杯边缘、餐巾纸）
2. 传递链条在物理上是否可行（毒素如何从施毒点最终进入受害者体内）
3. 现有证人陈述是否为该机制提供了支撑，或存在明显矛盾

你不知道正确答案，只能根据证据和逻辑判断推理是否自洽，若存在矛盾且侦探无法解释则不予通过。
分析完成后，必须调用 submit_verdict 提交你的裁决。"""

VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "提交对侦探作案过程推断的裁决",
        "parameters": {
            "type": "object",
            "properties": {
                "approved": {
                    "type": "boolean",
                    "description": "推断是否通过评审",
                },
                "feedback": {
                    "type": "string",
                    "description": (
                        "评审意见（3-5句）：说明通过或驳回的理由，"
                        "如驳回需指出具体缺失的证据或逻辑漏洞。"
                    ),
                },
            },
            "required": ["approved", "feedback"],
        },
    },
}


class JudgeAgent:
    def __init__(self, router: LLMRouter, case_description: str, evidence: dict):
        self.router           = router
        self.case_description = case_description
        self.evidence         = evidence

    def evaluate(self, investigation_log: list[dict], deduction: dict) -> dict:
        """
        Synchronously evaluate the detective's crime-process deduction.

        deduction: {"location": str, "process": str}
        Returns:   {"approved": bool, "feedback": str}
        """
        prompt   = self._build_prompt(investigation_log, deduction)
        messages = [{"role": "user", "content": prompt}]

        response = self.router.chat(
            messages=messages,
            tools=[VERDICT_TOOL],
            system=JUDGE_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.3,
        )

        if response.tool_calls:
            return response.tool_calls[0].input  # {"approved": bool, "feedback": str}

        # Fallback: no tool call — treat as approved with text response
        return {"approved": True, "feedback": response.content or "（评审员未提交结构化裁决）"}

    # ── Prompt building ──────────────────────────────────────────────────────

    def _build_prompt(self, investigation_log: list[dict], deduction: dict) -> str:
        evidence_text = "\n".join(
            f"  【{k}】{v}" for k, v in self.evidence.items()
        )

        if investigation_log:
            lines = []
            for entry in investigation_log:
                lines.append(f"  侦探问 {entry['npc']}：{entry['question']}")
                lines.append(f"  {entry['npc']} 答：{entry['answer']}\n")
            dialogue_text = "\n".join(lines)
        else:
            dialogue_text = "  （无对话记录）"

        return f"""## 案件背景
{self.case_description}

## 法医物证报告
{evidence_text}

## 侦探调查过程（证人问答记录）
{dialogue_text}

## 侦探提交的作案过程推断
- 毒素施加位置：{deduction.get('location', '未指定')}
- 作案过程描述：{deduction.get('process', '未提交')}

请分析这个推断是否能自洽地解释所有物证，然后调用 submit_verdict 提交裁决。"""
