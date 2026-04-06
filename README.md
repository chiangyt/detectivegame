# Detective Game

基于 LLM Agent 的交互式侦探推理游戏。审讯三位嫌疑人，收集线索，揭开真相。

## 快速开始

```bash
pip install -r requirements.txt
```

在项目根目录创建 `api_key.env`：

```
OPENAI_API_KEY=sk-...
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

## 技术

| 组件 | 实现 |
|------|------|
| LLM | OpenAI `gpt-4o-mini`，temperature=0.3，SSE 流式输出 |
| 后端 | Flask + `stream_with_context` |
| 前端 | 原生 HTML/CSS/JS，noir 复古风格，三栏布局 |
| 条件检测 | 关键词指纹匹配，零额外 API 开销 |
| NPC 知识 | 条件触发后动态 override，实时刷新 system prompt |

## 项目结构

```
Detective/
├── app.py                  # 入口 + Flask 服务器 + API 路由
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
