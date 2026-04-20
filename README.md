# Detective Game

基于 LLM Agent 的交互式侦探推理游戏。审讯三位嫌疑人，收集线索，揭开真相。

## 快速开始

```bash
pip install -r requirements.txt
```

在项目根目录创建 `api_key.env`：

```
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
CLAUDE_API_KEY=sk-ant-...
```

启动：

```bash
python app.py
```

浏览器访问 `http://localhost:5000`

---

## 玩法

1. **阅读案情摘要** — 点击粗体证物名称收集到证据室
2. **审讯嫌疑人** — 选择嫌疑人卡片，自由提问
3. **发现线索** — 问到关键信息时自动触发线索（共 5 条）
4. **最终指控** — 集齐全部线索后解锁，指出凶手、作案手法、动机

> 线索之间存在依赖关系，部分信息需按顺序触发才能获得。

---

## AI 侦探模式

点击「AI侦探自动调查」可进入观战模式，由 AI 自主审讯三位嫌疑人。支持通过 `?provider=` 参数切换侦探模型：

| Provider | 模型 |
|----------|------|
| `deepseek`（默认）| DeepSeek `deepseek-reasoner` |
| `claude` | Anthropic `claude-sonnet-4-6` |
| `openai` | OpenAI `gpt-4o` |

```
http://localhost:5000/api/auto-investigate?provider=deepseek
```

- 侦探通过 tool use 调用 `ask_npc`、`search_evidence`、`deduce_crime_process`、`make_accusation`
- 提交作案机制推断时，Claude 裁判会介入评审——推理不自洽则驳回并要求重新调查
- 发现新线索时，系统自动向侦探注入提示，引导调查方向
- 必须触发全部 5 条线索后，才能提交最终指控

---

## 技术

| 组件 | 实现 |
|------|------|
| NPC LLM | OpenAI `gpt-4o-mini`，temperature=0.3，SSE 流式输出 |
| AI 侦探 | 可切换：DeepSeek / Claude / OpenAI，通过 `LLMRouter` 统一接口 |
| 裁判 | Anthropic `claude-sonnet-4-6`，强制 tool use 返回结构化裁决 |
| 后端 | Flask + `stream_with_context`
| 前端 | 原生 HTML/CSS/JS，noir 复古风格，三栏布局 |
| 条件检测 | 关键词指纹匹配，零额外 API 开销 |
| NPC 知识 | 条件触发后动态 override，实时刷新 system prompt |

## 项目结构

```
Detective/
├── app.py                  # Flask 服务器 + API 路由 + provider 切换
├── llm_router.py           # 统一 OpenAI / Anthropic 接口
├── detective_agent.py      # AI 侦探 Agent（tool use 循环，多 provider）
├── judge_agent.py          # Claude 裁判 Agent（评审作案机制推断）
├── npc_agent.py            # 单个 NPC LLM Agent
├── npc_agent_manager.py    # 3 个 Agent 编排 + 条件检测
├── condition_detector.py   # 关键词触发检测
├── script_manager.py       # 剧本数据 + override 系统
├── evidence_handler.py     # 证据查询
├── scripts/                # 案件数据（JSON）
│   ├── case.json
│   ├── jordan.json
│   ├── morgan.json
│   ├── lucas.json
│   └── triggers.json
├── templates/index.html    # Web UI
└── requirements.txt
```
