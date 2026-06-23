.PHONY: help install train clean lint format

# 默认目标
help:
	@echo "WAD - Web Attack Detection System"
	@echo ""
	@echo "使用方法:"
	@echo "  make [target]"
	@echo ""
	@echo "目标:"
	@echo "  help            显示此帮助信息"
	@echo "  install         安装所有依赖"
	@echo "  train           训练模型"
	@echo "  clean           清理临时文件"
	@echo "  lint            代码检查"
	@echo "  format          代码格式化"
	@echo "  start           启动所有服务"
	@echo "  start-inference 启动推理服务"
	@echo "  start-web       启动 Web 前端"
	@echo "  stop            停止所有服务"

# 安装依赖
install:
	@echo "安装训练模块依赖..."
	cd Train_Model && pip install -r requirements.txt
	@echo "安装 Web 前端依赖..."
	cd Web_Frontend && pip install -r requirements.txt
	@echo "安装推理服务依赖..."
	cd Inference_API && pip install -r requirements.txt
	@echo "依赖安装完成！"

# 训练模型
train:
	@echo "开始训练模型..."
	cd Train_Model && python train.py
	@echo "模型训练完成！"

# 清理临时文件
clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .mypy_cache/
	@echo "清理完成！"

# 代码检查
lint:
	@echo "运行代码检查..."
	flake8 Train_Model/ Web_Frontend/ Inference_API/ --max-line-length=127 --statistics
	@echo "代码检查完成！"

# 代码格式化
format:
	@echo "格式化代码..."
	black Train_Model/ Web_Frontend/ Inference_API/ --line-length=127
	@echo "代码格式化完成！"

# 启动所有服务
start:
	@echo "启动所有服务..."
	@echo "启动推理服务..."
	cd Inference_API && python api_server.py &
	@echo "等待推理服务启动..."
	sleep 5
	@echo "启动 Web 前端..."
	cd Web_Frontend && python app.py

# 启动推理服务
start-inference:
	@echo "启动推理服务..."
	cd Inference_API && python api_server.py

# 启动 Web 前端
start-web:
	@echo "启动 Web 前端..."
	cd Web_Frontend && python app.py

# 停止所有服务
stop:
	@echo "停止所有服务..."
	pkill -f "python api_server.py" || true
	pkill -f "python app.py" || true
	@echo "服务已停止！"

# 生成文档
docs:
	@echo "生成文档..."
	cd docs && make html
	@echo "文档生成完成！"

# 部署
deploy:
	@echo "部署项目..."
	# 这里可以添加部署脚本
	@echo "部署完成！"

# 备份
backup:
	@echo "备份项目..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude='.git' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.idea' \
		--exclude='venv' \
		--exclude='*.egg-info' \
		.
	@echo "备份完成！"

# 更新依赖
update-deps:
	@echo "更新依赖..."
	cd Train_Model && pip install --upgrade -r requirements.txt
	cd Web_Frontend && pip install --upgrade -r requirements.txt
	cd Inference_API && pip install --upgrade -r requirements.txt
	@echo "依赖更新完成！"

# 检查依赖安全
security-check:
	@echo "检查依赖安全..."
	pip install safety
	safety check -r Train_Model/requirements.txt
	safety check -r Web_Frontend/requirements.txt
	safety check -r Inference_API/requirements.txt
	@echo "安全检查完成！"

# 生成 requirements.txt
freeze:
	@echo "生成 requirements.txt..."
	cd Train_Model && pip freeze > requirements.txt
	cd Web_Frontend && pip freeze > requirements.txt
	cd Inference_API && pip freeze > requirements.txt
	@echo "requirements.txt 生成完成！"