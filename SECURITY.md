# 安全政策

## 支持的版本

| 版本 | 支持状态 |
| ---- | -------- |
| 1.0.x | ✅ 支持 |
| < 1.0 | ❌ 不支持 |

---

## 报告漏洞

如果你发现安全漏洞，请**不要**通过公开的 Issue 报告。相反，请通过以下方式联系我们：

### 报告方式

1. **邮件报告**（推荐）
   - 发送邮件至：1141606412@qq.com
   - 邮件标题：`[WAD Security] 漏洞报告`

2. **GitHub Security Advisories**
   - 访问项目的 Security 页面
   - 点击 "Report a vulnerability"

### 报告内容

请包含以下信息：

```markdown
**漏洞类型**
例如：SQL注入、XSS、命令注入、路径遍历等

**漏洞描述**
详细描述漏洞的性质和影响

**复现步骤**
1. 步骤一
2. 步骤二
3. ...

**影响范围**
描述漏洞可能造成的危害

**建议修复方案**（可选）
如果你有修复建议，请一并提供

**环境信息**
- 操作系统：
- Python版本：
- 项目版本：
```

### 响应时间

- **初始响应**：48小时内
- **漏洞确认**：7个工作日内
- **修复发布**：根据严重程度，通常在30天内

---

## 安全最佳实践

### 部署安全

1. **HTTPS**
   ```nginx
   server {
       listen 443 ssl;
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       # ... 其他配置
   }
   ```

2. **防火墙**
   ```bash
   # 只开放必要端口
   ufw allow 80/tcp
   ufw allow 443/tcp
   ufw enable
   ```

3. **环境变量**
   ```bash
   # 不要在代码中硬编码敏感信息
   export API_KEY="your-secret-key"
   export DATABASE_URL="sqlite:///path/to/db.sqlite"
   ```

### API 安全

1. **输入验证**
   - 所有用户输入都应进行验证和清理
   - 使用参数化查询防止SQL注入
   - 对输出进行编码防止XSS

2. **速率限制**
   ```python
   from flask_limiter import Limiter
   
   limiter = Limiter(app, key_func=get_remote_address)
   
   @app.route('/api/detect', methods=['POST'])
   @limiter.limit("10 per minute")
   def detect():
       # ...
   ```

3. **CORS 配置**
   ```python
   from flask_cors import CORS
   
   # 只允许可信域名
   CORS(app, resources={
       r"/api/*": {"origins": ["https://yourdomain.com"]}
   })
   ```

### 数据安全

1. **数据库安全**
   - 定期备份数据库
   - 使用加密连接
   - 限制数据库访问权限

2. **日志安全**
   - 不要记录敏感信息
   - 定期轮转日志文件
   - 限制日志访问权限

3. **模型安全**
   - 验证模型文件来源
   - 定期更新模型
   - 监控模型性能

---

## 安全更新

### 获取安全更新

1. **Watch 本项目**
   - 在 GitHub 上 Watch 本项目
   - 启用安全公告通知

2. **检查更新**
   ```bash
   # 拉取最新代码
   git pull origin main
   
   # 检查版本
   git log --oneline -10
   ```

3. **应用更新**
   ```bash
   # 备份当前版本
   cp -r /path/to/project /path/to/project.backup
   
   # 更新代码
   git pull origin main
   
   # 更新依赖
   pip install -r requirements.txt
   
   # 重启服务
   systemctl restart wad
   ```

---

## 安全配置清单

### 生产环境部署前检查

- [ ] 启用 HTTPS
- [ ] 配置防火墙
- [ ] 设置环境变量
- [ ] 配置速率限制
- [ ] 设置 CORS 策略
- [ ] 配置日志记录
- [ ] 设置数据库备份
- [ ] 配置监控告警
- [ ] 更新所有依赖
- [ ] 禁用调试模式

### 定期安全检查

- [ ] 检查依赖漏洞（每月）
- [ ] 审查访问日志（每周）
- [ ] 更新系统补丁（每月）
- [ ] 备份数据验证（每月）
- [ ] 安全配置审查（每季度）

---

## 依赖安全

### 检查依赖漏洞

```bash
# 使用 safety 检查
pip install safety
safety check

# 使用 pip-audit
pip install pip-audit
pip-audit

# 使用 bandit 检查代码安全
pip install bandit
bandit -r Web_Frontend/
```

### 更新依赖

```bash
# 查看过时的包
pip list --outdated

# 更新特定包
pip install --upgrade package-name

# 更新所有依赖（谨慎使用）
pip install --upgrade -r requirements.txt
```

---

## 安全联系人

如果你有任何安全相关的问题或建议，请联系：

- **邮箱**：1141606412@qq.com
- **GitHub**：Chesapeake-Ripper

---

## 致谢

感谢所有帮助改进项目安全性的贡献者！

---

## 许可证

本安全政策采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可证。