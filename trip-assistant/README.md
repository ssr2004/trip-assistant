# TravelMind：旅行AI助手

基于 LangGraph 的多 Agent 智能旅行规划系统，面向“从自然语言需求到结构化旅行方案生成”的完整业务闭环。

项目目标不是做一个简单聊天机器人，而是构建一个具备 Agent 能力的旅行规划系统：能够识别用户意图、抽取旅行实体、规划任务、调用工具、检索攻略和政策知识、记忆用户偏好，并生成可解释的旅行方案。

## 核心能力

- **自然语言需求解析**：识别旅行规划、航班搜索、酒店搜索、景点查询、政策问答等意图。
- **任务规划与执行**：将复杂旅行需求拆解为航班、酒店、景点、攻略、行程生成等任务。
- **多工具协调**：通过统一 Tool Registry 管理航班、酒店、景点、政策、攻略和行程生成工具。
- **标准工具协议**：工具统一返回 `success/data/error/metadata`，便于 Agent 识别成功、失败和数据来源。
- **RAG 检索增强**：支持政策、攻略、FAQ 文档检索，并在回答中保留来源引用。
- **用户记忆系统**：保存短期会话、长期偏好和历史旅行场景，用于个性化推荐。
- **结构化旅行方案**：输出出行概览、航班推荐、酒店推荐、景点推荐、每日行程、预算估算和注意事项。
- **工程化落地**：提供 FastAPI 接口、WebSocket 对话、Docker 部署和自动化测试。

## 技术栈

| 模块 | 技术 |
|---|---|
| Agent 编排 | LangGraph |
| LLM 接入 | DeepSeek / GLM / Qwen / OpenAI 兼容接口 |
| RAG | LangChain + FAISS / Chroma |
| Web 后端 | FastAPI |
| 前端方向 | Vue 3 + Element Plus |
| 数据库 | SQLite / PostgreSQL |
| 配置管理 | pydantic-settings + `.env` |
| 测试 | pytest + pytest-asyncio |
| 部署 | Docker / Docker Compose |

## 环境要求

当前开发验证环境：

```text
Python 3.13.7
```

依赖文件已按 Python 3.13 兼容性调整。若后续接入某些只支持低版本 Python 的向量库或深度学习组件，也可以选择 Python 3.11 作为部署环境。

## 项目结构

```text
trip-assistant/
├── app/                    # FastAPI 应用入口和接口
│   ├── main.py
│   └── config.py
├── core/                   # Agent 核心流程
│   ├── agent.py            # 主 Agent 工作流
│   ├── intent.py           # 意图识别与实体抽取
│   ├── planner.py          # 任务规划器
│   ├── response_builder.py # 工具结果聚合和最终回复生成
│   ├── state.py            # LangGraph 状态定义
│   └── memory/             # 记忆系统实现
├── database/               # 数据库初始化和后续迁移脚本
├── docs/                   # 正式项目文档，进入 Git 管理
├── frontend/               # 前端展示，后续升级为 Vue 3 + Element Plus
├── models/                 # Pydantic 数据模型
├── rag/                    # RAG 检索系统
│   └── documents/          # 政策、攻略等知识库文档
├── tests/                  # 自动化测试
├── tools/                  # 工具注册表和业务工具
├── .dockerignore           # Docker 构建忽略规则
├── .env.example            # 环境变量示例，不包含真实密钥
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

> `local-docs/` 用于存放本地方案、模块说明和开发草稿，已在 `.gitignore` 中排除，不进入 Git 管理。

## 快速开始

### 1. 进入项目目录

```powershell
cd trip-assistant
```

### 2. 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv --prompt trip-assistant .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
python -m venv --prompt trip-assistant .venv
source .venv/bin/activate
```

### 3. 安装依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果只想检查当前环境依赖是否存在冲突：

```powershell
python -m pip check
```

### 4. 配置环境变量

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

然后根据实际情况填写 `.env`。不要将真实 API Key 写入代码或提交到 Git。

当前 `.env.example` 已预留：

```text
LLM_API_KEY
LLM_BASE_URL
EMBEDDING_API_KEY
EMBEDDING_BASE_URL
AMADEUS_API_KEY
AMADEUS_API_SECRET
AMAP_API_KEY
WEATHER_API_KEY
DATABASE_URL
REDIS_URL
```

### 5. 启动后端服务

```powershell
python -m uvicorn app.main:app --reload
```

服务启动后访问：

```text
http://localhost:8000/
```

## Docker 启动

启动前先准备本地 `.env`，真实 Key 只写入本地 `.env`，不要提交：

```bash
cp .env.example .env
docker-compose up -d
```

服务默认暴露：

```text
http://localhost:8000/
```

`docker-compose.yml` 会将 `./data` 挂载到容器 `/app/data`，用于 SQLite、长期记忆和运行历史等本地数据。Redis 仅在 compose 内部网络暴露给后端服务，不默认映射到宿主机。

## API 示例

### 健康检查

```bash
curl http://localhost:8000/
```

### 对话接口

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我要从郑州去杭州玩三天，预算3000，6月10日出发"}'
```

### WebSocket 对话

```text
ws://localhost:8000/ws/chat
```

## 测试

```powershell
python -m pytest
```

完整本地质量门禁：

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_gate.py
```

需要机器可读结果时：

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_gate.py --json-compact
```

当前已覆盖：

- 意图识别与实体抽取；
- 智能任务规划；
- 工具注册和工具执行；
- 工具标准返回结构；
- Agent 最终响应聚合；
- 完整 Agent 冒烟流程。

## 数据与密钥策略

- 开发阶段优先使用模拟数据和本地文档，保证 Agent 闭环稳定。
- 后续可接入 Amadeus、高德地图等合规开放 API。
- 不建议爬取携程、飞猪、去哪儿等平台数据。
- 所有真实 API Key 通过 `.env` 管理，严禁硬编码到代码中。
- `.env`、`.venv`、运行时数据和本地开发文档不进入 Git。

## 开发规范

本项目按业务功能模块逐步开发：

1. 一个业务功能模块一个一个实现。
2. 每完成一个模块，补充必要测试。
3. 每完成一个模块，写一份模块说明文档到 `local-docs/`。
4. 每完成一个模块，提交一次 Git。
5. Commit message 使用中文，简洁明确。
6. 提交后推送到远程仓库。

模块说明文档建议包含：

```text
1. 模块名称
2. 模块目标
3. 改动文件
4. 实现思路
5. 技术细节
6. 测试方式
7. Git commit 信息
8. 后续可优化点
```

## 当前开发路线

已完成：

1. 项目骨架治理与开发规范整理。
2. 旅行需求解析模块：意图识别、实体抽取、缺失信息追问。
3. 智能任务规划模块：将旅行需求拆解为可执行任务。
4. 旅行工具补齐：政策检索、攻略检索、行程生成。
5. Agent 响应生成：聚合航班、酒店、景点、攻略和行程结果。
6. 工具返回结构统一：统一 `success/data/error/metadata` 协议。
7. 配置治理与依赖兼容性修复。

后续计划：

1. 抽象统一工具结果模型和工具基类辅助方法。
2. 接入 LLM，增强意图解析、任务规划和最终回复生成。
3. 升级 RAG：向量检索、混合检索、重排序和 grounded answer。
4. 接入 Memory 用户画像：偏好提取、长期记忆和个性化推荐。
5. 接入天气、地图和合规真实 API。
6. 开发 Vue 3 + Element Plus 前端展示。

## 许可证

MIT License
