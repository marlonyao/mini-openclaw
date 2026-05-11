# mini-openclaw 设计文档（Python 版）

一个简化版的 OpenClaw，用 Python + asyncio 实现，TDD 驱动开发。

## 核心特性（参考 OpenClaw 源码）

### Phase 1: 配置系统 + 模型管理
- **Config**: JSON/YAML 配置文件加载（参考 `config.js`）
- **Models**: 多 provider 模型注册、API key 管理（参考 `models.json`）
- **Model Registry**: 模型发现、provider 解析（参考 `model-catalog.runtime.js`）
- **Provider HTTP 客户端**: OpenAI-compatible + Anthropic-compatible API（streaming 支持）

### Phase 2: 会话管理
- **Session**: 会话创建、消息管理（JSONL 格式，参考 `SessionManager`）
- **Session Store**: 会话持久化、过期清理
- **Session Router**: 按 channel + peer 路由消息到正确的会话

### Phase 3: Agent Loop
- **Agent Loop**: 接收消息 → 组装上下文 → LLM 推理 → 工具执行 → 流式回复
- **Tool System**: 工具注册、schema 定义、执行框架
- **Streaming**: 流式输出

### Phase 4: 上下文管理
- **Compaction**: 长对话自动压缩/摘要
- **Session Pruning**: 旧工具结果裁剪
- **Context Assembly**: 上下文组装

### Phase 5: 记忆系统
- **Memory**: 长期记忆 + 每日笔记
- **Memory Search**: 关键词搜索

### Phase 6: 系统 Prompt + Skills
- **System Prompt Builder**: 动态系统 prompt 组装
- **Bootstrap Files**: AGENTS.md/SOUL.md/USER.md 等注入
- **Skills**: 技能发现和加载

### Phase 7: Gateway + Channel
- **Gateway**: WebSocket 网关服务器
- **Channel**: 消息通道抽象
- **Queue**: 命令队列

### Phase 8: 高级特性
- **Sub-agents**: 子 agent
- **Model Failover**: 模型故障转移

## 技术栈
- Python 3.11+
- asyncio（异步运行时）
- pydantic（配置和模型验证）
- aiohttp / httpx（HTTP 客户端，用于 LLM API）
- websockets（WebSocket 服务器）
- aiofile（异步文件 I/O）

## 项目结构
```
mini-openclaw/
├── pyproject.toml
├── DESIGN.md
├── src/
│   └── mini_openclaw/
│       ├── __init__.py
│       ├── main.py
│       ├── config/          # Phase 1
│       ├── models/          # Phase 1
│       ├── session/         # Phase 2
│       ├── agent/           # Phase 3
│       ├── tools/           # Phase 3
│       ├── context/         # Phase 4
│       ├── memory/          # Phase 5
│       ├── prompt/          # Phase 6
│       ├── skills/          # Phase 6
│       ├── gateway/         # Phase 7
│       └── channel/         # Phase 7
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_models.py
    ├── test_session.py
    └── ...
```
