# TravelMind：旅行AI助手

基于 LangGraph 的多 Agent 智能旅行规划系统，面向“从自然语言需求到结构化旅行方案生成”的完整业务闭环。

项目目标不是做一个简单聊天机器人，而是构建一个具备 Agent 能力的旅行规划系统：能够识别用户意图、抽取旅行实体、规划任务、调用工具、检索攻略和政策知识、记忆用户偏好，并生成可解释的旅行方案。

## 核心能力规划

- **自然语言需求解析**：识别旅行规划、航班搜索、酒店搜索、景点查询、政策问答等意图。
- **任务规划与执行**：将复杂旅行需求拆解为航班、酒店、景点、攻略、行程生成等任务。
- **多工具协调**：通过统一 Tool Registry 管理航班、酒店、景点、天气、地图、RAG 等工具。
- **RAG 检索增强**：支持政策、攻略、FAQ 文档检索，并在回答中保留来源引用。
- **用户记忆系统**：保存短期会话、长期偏好和历史旅行场景，用于个性化推荐。
- **结构化旅行方案**：输出航班推荐、酒店推荐、每日行程、预算估算和注意事项。
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
│   ├── executor.py         # 任务执行器
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

```bash
cd trip-assistant
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

然后根据实际情况填写 `.env`。不要将真实 API Key 写入代码或提交到 Git。

### 5. 启动后端服务

```bash
uvicorn app.main:app --reload
```

或：

```bash
python -m app.main
```

服务启动后访问：

```text
http://localhost:8000/
```

## Docker 启动

```bash
docker-compose up -d
```

## API 示例

### 健康检查

```bash
curl http://localhost:8000/
```

### 对话接口

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我要从郑州去杭州玩三天，预算3000"}'
```

### WebSocket 对话

```text
ws://localhost:8000/ws/chat
```

## 测试

```bash
pytest
```

如果当前环境尚未安装依赖，请先执行：

```bash
pip install -r requirements.txt
```

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

## 数据与密钥策略

- 开发阶段优先使用模拟数据和种子数据，保证 Agent 闭环稳定。
- 后续可接入 Amadeus、高德地图等合规开放 API。
- 不建议爬取携程、飞猪、去哪儿等平台数据。
- 所有真实 API Key 通过 `.env` 管理，严禁硬编码到代码中。

## 当前开发路线

1. 项目骨架治理与开发规范整理。
2. 旅行需求解析模块：意图识别、实体抽取、缺失信息追问。
3. 任务规划模块：将旅行需求拆解为可执行任务。
4. 工具调用模块：航班、酒店、景点、攻略、天气等工具。
5. 行程生成模块：生成结构化旅行方案。
6. RAG 检索模块：政策与攻略知识检索。
7. Memory 用户画像模块：提取并使用用户偏好。
8. 前端展示模块：Vue 3 + Element Plus 展示旅行方案。

## 许可证

MIT License
