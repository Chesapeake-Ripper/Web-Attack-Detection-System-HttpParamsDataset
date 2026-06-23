# WAD Web漏洞攻击检测 API - Postman 调用文档

## API 基础信息

| 项目 | 值 |
|------|-----|
| 服务地址 | `http://localhost:8000` |
| 协议 | HTTP |
| 数据格式 | JSON |
| 编码 | UTF-8 |

---

```
http://localhost:9000/health
```

## 接口一：健康检查

### 请求配置

| 配置项 | 值 |
|--------|-----|
| Method | `GET` |
| URL | `{{base_url}}/health` |

### Postman 操作步骤

1. 新建请求 → 选择 `GET`
2. 输入 URL：`http://localhost:8000/health`
3. 点击 **Send**

### 响应示例（200 OK）

```json
{
  "status": "ok",
  "service": "WAD API",
  "version": "1.0.0"
}
```

---

## 接口二：单条 Payload 检测

### 请求配置

| 配置项 | 值 |
|--------|-----|
| Method | `POST` |
| URL | `{{base_url}}/predict` |
| Content-Type | `application/json` |

### 请求 Body（raw JSON）

```json
{
  "payload": "' OR 1=1 --",
  "model": "lgbm"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `payload` | string | 是 | HTTP 请求参数值，待检测的字符串 |
| `model` | string | 否 | 模型选择：`lgbm`（默认）或 `textcnn` |

### Postman 操作步骤

1. 新建请求 → 选择 `POST`
2. 输入 URL：`http://localhost:8000/predict`
3. 点击 **Body** → 选择 **raw** → 下拉选择 **JSON**
4. 粘贴请求体
5. 点击 **Send**

### 响应示例（200 OK）

```
{
    "success": true,
    "result": {
        "payload": "' OR 1=1 --",
        "label": "sqli",
        "label_cn": "SQL注入",
        "confidence": 0.997048,
        "all_probs": {
            "cmdi": 0.000169,
            "norm": 0.002718,
            "path-traversal": 4e-05,
            "sqli": 0.997048,
            "xss": 2.6e-05
        },
        "model": "lgbm"
    },
    "elapsed_ms": 345.35
}

  {
    "success": true,
    "result": {
      "payload": "' OR 1=1 --",
      "label": "sqli",
      "label_cn": "SQL注入",
      "confidence": 0.9970,
      "all_probs": {
        "cmdi": 0.0002,
        "norm": 0.0027,
        "path-traversal": 0.0000,
        "sqli": 0.9970,
        "xss": 0.0000
      },
      "model": "lgbm"
    },
    "elapsed_ms": 345.35
  }

```



```json
{
  "success": true,
  "result": {
    "payload": "' OR 1=1 --",
    "label": "sqli",
    "label_cn": "SQL注入",
    "confidence": 0.9952,
    "all_probs": {
      "norm": 0.001,
      "sqli": 0.995,
      "xss": 0.002,
      "cmdi": 0.001,
      "path-traversal": 0.001
    },
    "model": "lgbm"
  },
  "elapsed_ms": 12.35
}
```

### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | bool | 请求是否成功 |
| `result.label` | string | 预测标签：`norm`/`sqli`/`xss`/`cmdi`/`path-traversal` |
| `result.label_cn` | string | 中文标签 |
| `result.confidence` | float | 最高类别置信度（0~1） |
| `result.all_probs` | object | 全部 5 个类别的概率分布 |
| `elapsed_ms` | float | 推理耗时（毫秒） |

---

## 接口三：批量 Payload 检测

### 请求配置

| 配置项 | 值 |
|--------|-----|
| Method | `POST` |
| URL | `{{base_url}}/predict/batch` |
| Content-Type | `application/json` |

### 请求 Body（raw JSON）

```json
{
  "payloads": [
    "' OR 1=1 --",
    "<script>alert(1)</script>",
    "; cat /etc/passwd",
    "../../../../etc/shadow",
    "hello world"
  ],
  "model": "lgbm"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `payloads` | string[] | 是 | payload 列表，**最多 500 条** |
| `model` | string | 否 | 模型选择：`lgbm`（默认）或 `textcnn` |

### Postman 操作步骤

1. 新建请求 → 选择 `POST`
2. 输入 URL：`http://localhost:8000/predict/batch`
3. 点击 **Body** → 选择 **raw** → 下拉选择 **JSON**
4. 粘贴请求体
5. 点击 **Send**

### 响应示例（200 OK）

```json
{
  "success": true,
  "total": 5,
  "attack_count": 4,
  "normal_count": 1,
  "elapsed_ms": 25.67,
  "results": [
    {
      "payload": "' OR 1=1 --",
      "label": "sqli",
      "label_cn": "SQL注入",
      "confidence": 0.9952,
      "all_probs": {
        "norm": 0.001,
        "sqli": 0.995,
        "xss": 0.002,
        "cmdi": 0.001,
        "path-traversal": 0.001
      },
      "model": "lgbm"
    },
    {
      "payload": "<script>alert(1)</script>",
      "label": "xss",
      "label_cn": "XSS攻击",
      "confidence": 0.9821,
      "all_probs": {
        "norm": 0.002,
        "sqli": 0.003,
        "xss": 0.982,
        "cmdi": 0.005,
        "path-traversal": 0.008
      },
      "model": "lgbm"
    }
  ]
}
```

### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | bool | 请求是否成功 |
| `total` | int | 总检测条数 |
| `attack_count` | int | 攻击条数 |
| `normal_count` | int | 正常条数 |
| `elapsed_ms` | float | 总推理耗时（毫秒） |
| `results` | array | 每条检测结果（结构同单条响应） |

---

## 错误响应

| HTTP 状态码 | 场景 | 示例 |
|-------------|------|------|
| `400` | 参数校验失败 | `{"detail": "payload 不能为空"}` |
| `400` | model 参数错误 | `{"detail": "model 须为 lgbm 或 textcnn"}` |
| `400` | 超过500条限制 | `{"detail": "单次最多 500 条"}` |
| `500` | 推理异常 | `{"detail": "推理失败：..."}` |

---

## Postman 环境变量配置（推荐）

在 Postman 中设置 **Environment Variables**：

| Variable | Initial Value |
|----------|---------------|
| `base_url` | `http://localhost:8000` |

然后请求 URL 统一写为 `{{base_url}}/predict`，方便切换环境。

---

## 攻击类型速查表

| label | label_cn | 说明 |
|-------|----------|------|
| `norm` | 正常流量 | 无攻击特征 |
| `sqli` | SQL注入 | SQL 语句注入 |
| `xss` | XSS攻击 | 跨站脚本攻击 |
| `cmdi` | 命令注入 | 系统命令注入 |
| `path-traversal` | 路径穿越 | 目录遍历攻击 |
