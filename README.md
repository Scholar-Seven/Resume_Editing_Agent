# 简历优化助手

一个基于 Flask 的应用：上传简历与岗位 JD，调用 DeepSeek 进行分析，生成建议并输出优化后的 DOCX。

## 运行要求

- Docker（推荐）
- DeepSeek API Key

程序通过环境变量读取 `DEEPSEEK_API_KEY`。缺失时服务会直接启动失败。

## 快速开始（Docker Compose）

1. 修改 Resume_Editing_Agent/docker-compose.yml 环境变量参数：

```
DEEPSEEK_API_KEY= 你的API 
```

2. 构建并启动：

```
docker compose up -d --build
```

3. 打开页面：

```
http://localhost:5000
```

## API

该 API 是有状态的，使用内存存储。推荐调用流程：

1. `POST /upload` 上传简历与 JD
2. `POST /analyze` 生成分析报告
3. `POST /confirm` 确认要应用的建议
4. `POST /optimize` 生成优化后的简历
5. `GET /download/<resume_id>` 下载文件

### 1) Upload

```
curl -F "resume_file=@/path/to/resume.pdf" \
     -F "jd_text=岗位描述文本" \
     http://localhost:5000/upload
```

返回：

```
{"resume_id":"..."}
```

### 2) Analyze

```
curl -H "Content-Type: application/json" \
     -d '{"resume_id":"<resume_id>"}' \
     http://localhost:5000/analyze
```

### 3) Confirm Suggestions

```
curl -H "Content-Type: application/json" \
     -d '{"resume_id":"<resume_id>","confirmed_suggestion_ids":["S1","S2"]}' \
     http://localhost:5000/confirm
```

### 4) Optimize

```
curl -H "Content-Type: application/json" \
     -d '{"resume_id":"<resume_id>"}' \
     http://localhost:5000/optimize
```

返回：

```
{"ok":true,"download_url":"/download/<resume_id>"}
```

### 5) Download

```
curl -O http://localhost:5000/download/<resume_id>
```

## 说明

- 内存存储是进程内的。生产环境建议替换为 Redis/SQLite，避免多 worker 场景下出现 `resume_id not found`。
- 大请求可能耗时较长，容器内 Gunicorn 已配置更长超时（见 `Dockerfile`）。
