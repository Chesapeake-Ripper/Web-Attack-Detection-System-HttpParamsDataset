# WAD API 文档

## 概述

WAD (Web Attack Detection) 系统提供两套 API 接口：

1. **Web 前端 API**（端口 5000）：供浏览器访问的完整功能接口
2. **推理服务 API**（端口 9000）：供内部调用的模型推理接口

---

## 一、Web 前端 API

### 1.1 系统状态

**请求**
```http
GET /api/status
```

**响应**
```json
{
  "success": true,
  "data": {
    "api_available": true,
    "models": {
      "lgbm": true,
      "textcnn": true
    },
    "database": "connected",
    "version": "1.0.0"
  }
}
```

### 1.2 单条检测

**请求**
```http
POST /api/detect
Content-Type: application/json

{
  "payload": "' OR 1=1 --",
  "model": "lgbm"
}
```

**参数说明**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| payload | string | 是 | 待检测的 HTTP 参数值 |
| model | string | 否 | 模型选择：`lgbm`（默认）或 `textcnn` |

**响应**
```json
{
  "success": true,
  "data": {
    "label": "sqli",
    "label_cn": "SQL注入",
    "confidence": 0.9997,
    "icon": "🔴",
    "risk": 3,
    "all_probs": {
      "norm": 0.0002,
      "sqli": 0.9997,
      "xss": 0.0001,
      "cmdi": 0.0000,
      "path-traversal": 0.0000
    }
  }
}
```

### 1.3 批量检测

**请求**
```http
POST /api/detect/batch
Content-Type: application/json

{
  "payloads": [
    "' OR 1=1 --",
    "<script>alert(1)</script>",
    "normal text"
  ],
  "model": "lgbm"
}
```

**参数说明**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| payloads | array | 是 | Payload 列表，最多 500 条 |
| model | string | 否 | 模型选择：`lgbm`（默认）或 `textcnn` |

**响应**
```json
{
  "success": true,
  "data": {
    "total": 3,
    "attack_count": 2,
    "normal_count": 1,
    "results": [
      {
        "payload": "' OR 1=1 --",
        "label": "sqli",
        "label_cn": "SQL注入",
        "confidence": 0.9997
      },
      // ...
    ]
  }
}
```

### 1.4 双模型对比

**请求**
```http
POST /api/compare
Content-Type: application/json

{
  "payload": "' OR 1=1 --"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "lgbm": {
      "label": "sqli",
      "confidence": 0.9997
    },
    "textcnn": {
      "label": "sqli",
      "confidence": 0.9995
    },
    "consistent": true
  }
}
```

### 1.5 AI 深度分析

**请求**
```http
POST /api/analyze
Content-Type: application/json

{
  "payload": "' OR 1=1 --",
  "detection_result": {
    "label": "sqli",
    "confidence": 0.9997
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "analysis": "这是一个典型的 SQL 注入攻击载荷...",
    "defense_suggestions": [
      "使用参数化查询",
      "输入验证和过滤",
      "最小权限原则"
    ]
  }
}
```

### 1.6 统计数据

**请求**
```http
GET /api/stats
```

**响应**
```json
{
  "success": true,
  "data": {
    "total_detections": 1234,
    "attack_detections": 567,
    "normal_detections": 667,
    "attack_rate": 0.459,
    "daily_stats": [
      {"date": "2024-01-01", "count": 123},
      // ...
    ],
    "model_usage": {
      "lgbm": 800,
      "textcnn": 434
    }
  }
}
```

### 1.7 历史记录

**请求**
```http
GET /api/records?page=1&per_page=20&model=lgbm&type=sqli
```

**参数说明**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | int | 否 | 页码，默认 1 |
| per_page | int | 否 | 每页数量，默认 20 |
| model | string | 否 | 模型筛选 |
| type | string | 否 | 攻击类型筛选 |

**响应**
```json
{
  "success": true,
  "data": {
    "total": 1234,
    "page": 1,
    "per_page": 20,
    "records": [
      {
        "id": 1,
        "payload": "' OR 1=1 --",
        "label": "sqli",
        "confidence": 0.9997,
        "model": "lgbm",
        "created_at": "2024-01-01T12:00:00"
      },
      // ...
    ]
  }
}
```

---

## 二、推理服务 API

### 2.1 健康检查

**请求**
```http
GET /health
```

**响应**
```json
{
  "status": "ok",
  "service": "WAD API",
  "version": "1.0.0"
}
```

### 2.2 单条推理

**请求**
```http
POST /predict
Content-Type: application/json

{
  "payload": "' OR 1=1 --",
  "model": "lgbm"
}
```

**参数说明**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| payload | string | 是 | 待检测的 HTTP 参数值 |
| model | string | 否 | 模型选择：`lgbm`（默认）或 `textcnn` |

**响应**
```json
{
  "success": true,
  "result": {
    "payload": "' OR 1=1 --",
    "label": "sqli",
    "label_cn": "SQL注入",
    "confidence": 0.9997,
    "all_probs": {
      "norm": 0.0002,
      "sqli": 0.9997,
      "xss": 0.0001,
      "cmdi": 0.0000,
      "path-traversal": 0.0000
    },
    "model": "lgbm"
  },
  "elapsed_ms": 5.23
}
```

### 2.3 批量推理

**请求**
```http
POST /predict/batch
Content-Type: application/json

{
  "payloads": [
    "' OR 1=1 --",
    "<script>alert(1)</script>",
    "normal text"
  ],
  "model": "lgbm"
}
```

**参数说明**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| payloads | array | 是 | Payload 列表，最多 500 条 |
| model | string | 否 | 模型选择：`lgbm`（默认）或 `textcnn` |

**响应**
```json
{
  "success": true,
  "total": 3,
  "attack_count": 2,
  "normal_count": 1,
  "elapsed_ms": 15.67,
  "results": [
    {
      "payload": "' OR 1=1 --",
      "label": "sqli",
      "label_cn": "SQL注入",
      "confidence": 0.9997,
      "all_probs": {
        "norm": 0.0002,
        "sqli": 0.9997,
        "xss": 0.0001,
        "cmdi": 0.0000,
        "path-traversal": 0.0000
      },
      "model": "lgbm"
    },
    // ...
  ]
}
```

---

## 三、错误处理

### 3.1 错误响应格式

```json
{
  "success": false,
  "error": "错误信息",
  "detail": "详细错误描述（可选）"
}
```

### 3.2 常见错误码

| HTTP 状态码 | 说明 |
|------------|------|
| 400 | 请求参数错误 |
| 404 | 接口不存在 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

### 3.3 错误示例

**请求参数错误**
```json
{
  "success": false,
  "error": "payload 不能为空"
}
```

**模型不存在**
```json
{
  "success": false,
  "error": "model 须为 lgbm 或 textcnn"
}
```

**推理失败**
```json
{
  "success": false,
  "error": "推理失败",
  "detail": "模型加载失败，请检查模型文件"
}
```

---

## 四、使用示例

### 4.1 Python 示例

```python
import requests

# 单条检测
response = requests.post(
    "http://localhost:5000/api/detect",
    json={
        "payload": "' OR 1=1 --",
        "model": "lgbm"
    }
)
result = response.json()
print(f"检测结果: {result['data']['label_cn']}")
print(f"置信度: {result['data']['confidence']}")

# 批量检测
response = requests.post(
    "http://localhost:5000/api/detect/batch",
    json={
        "payloads": [
            "' OR 1=1 --",
            "<script>alert(1)</script>",
            "normal text"
        ],
        "model": "lgbm"
    }
)
result = response.json()
print(f"总检测数: {result['data']['total']}")
print(f"攻击数: {result['data']['attack_count']}")
```

### 4.2 cURL 示例

```bash
# 单条检测
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"payload": "'\'' OR 1=1 --", "model": "lgbm"}'

# 批量检测
curl -X POST http://localhost:5000/api/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"payloads": ["'\'' OR 1=1 --", "<script>alert(1)</script>"], "model": "lgbm"}'

# 健康检查
curl http://localhost:9000/health
```

### 4.3 JavaScript 示例

```javascript
// 单条检测
async function detect(payload) {
  const response = await fetch('http://localhost:5000/api/detect', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      payload: payload,
      model: 'lgbm'
    })
  });
  const result = await response.json();
  return result.data;
}

// 使用示例
detect("' OR 1=1 --").then(result => {
  console.log(`检测结果: ${result.label_cn}`);
  console.log(`置信度: ${result.confidence}`);
});
```

---

## 五、API 测试工具

### 5.1 Swagger UI

推理服务提供交互式 API 文档：

- 地址：http://localhost:9000/docs
- 功能：在线测试 API、查看请求/响应格式

### 5.2 ReDoc

推理服务提供 ReDoc 文档：

- 地址：http://localhost:9000/redoc
- 功能：更美观的 API 文档展示

### 5.3 Postman

可以使用 Postman 进行 API 测试：

1. 导入 API 文档
2. 创建请求集合
3. 配置环境变量
4. 运行测试

---

## 六、安全建议

### 6.1 生产环境

1. **HTTPS**：使用 HTTPS 加密通信
2. **认证**：添加 API 认证机制
3. **限流**：配置 API 限流策略
4. **CORS**：限制允许的域名
5. **输入验证**：严格验证输入参数

### 6.2 示例配置

```python
# Flask CORS 配置
from flask_cors import CORS

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdomain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# FastAPI CORS 配置
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 七、常见问题

### Q1: 如何切换模型？

在请求中指定 `model` 参数：
- `model: "lgbm"` - 使用 LightGBM 模型
- `model: "textcnn"` - 使用 TextCNN 模型

### Q2: 批量检测有数量限制吗？

是的，单次批量检测最多支持 500 条 Payload。

### Q3: 如何查看 API 文档？

推理服务提供 Swagger UI 文档：
- 地址：http://localhost:9000/docs

### Q4: 如何处理超时？

建议设置合理的超时时间：
- 单条检测：5 秒
- 批量检测：30 秒

### Q5: 如何优化性能？

1. 使用批量检测接口
2. 选择合适的模型（LightGBM 更快）
3. 部署多个推理服务实例
4. 使用缓存机制

---

## 八、更新日志

### v1.0.0 (2024-01-01)

- 初始版本发布
- 支持单条检测和批量检测
- 支持 LightGBM 和 TextCNN 模型
- 提供完整的 Web 界面