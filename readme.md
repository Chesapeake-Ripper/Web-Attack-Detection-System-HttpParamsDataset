## Web-Attack-Detection-System项目运行顺序：



```
python train.py → 生成 outputs/ 下所有模型文件

python app.py → 启动系统，访问 http://localhost:5000
```



[![image](https://private-user-images.githubusercontent.com/116621638/561555409-bceeae2e-9e87-48cd-9e21-5c1a5038bf02.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMjk1ODEsIm5iZiI6MTc3MzIyOTI4MSwicGF0aCI6Ii8xMTY2MjE2MzgvNTYxNTU1NDA5LWJjZWVhZTJlLTllODctNDhjZC05ZTIxLTVjMWE1MDM4YmYwMi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMxMVQxMTQxMjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yNDI5NTIzYjMwMDI5MGFjM2YwNzAzYTllNDU5ODQ5MWFlODA2YjIzM2ZkZGE4M2QyODFlM2I4Zjk0ZGI5NTQxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.ORlFSNm2o6SA0JPIcq6R_vHnlBoCHOTsQp8knwFqT34)](https://private-user-images.githubusercontent.com/116621638/561555409-bceeae2e-9e87-48cd-9e21-5c1a5038bf02.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzMyMjk1ODEsIm5iZiI6MTc3MzIyOTI4MSwicGF0aCI6Ii8xMTY2MjE2MzgvNTYxNTU1NDA5LWJjZWVhZTJlLTllODctNDhjZC05ZTIxLTVjMWE1MDM4YmYwMi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMxMVQxMTQxMjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yNDI5NTIzYjMwMDI5MGFjM2YwNzAzYTllNDU5ODQ5MWFlODA2YjIzM2ZkZGE4M2QyODFlM2I4Zjk0ZGI5NTQxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.ORlFSNm2o6SA0JPIcq6R_vHnlBoCHOTsQp8knwFqT34)

### 一、Python环境完整安装步骤（Windows Anaconda Prompt 中执行）



#### 1. 创建并激活虚拟环境



```conda
conda activate web_attack_detect
```



#### 2. 安装 PyTorch 核心框架（CUDA 12.0 适配）



```
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```



#### 3. 安装机器学习 & 数据处理核心包



```
pip install scikit-learn==1.3.2 lightgbm==4.6.0 imbalanced-learn==0.11.0 numpy==1.24.3 pandas==2.0.3 scipy==1.11.4 tqdm==4.66.4 joblib==1.4.2 openpyxl==3.1.5
```



#### 4. 安装深度学习 & 文本处理包



```
pip install transformers==4.41.2 torchtext==0.18.0 tensorboard==2.17.0
```



#### 5. 安装 Web 原型系统 & 数据库包



```
pip install Flask==3.1.3 Flask-SQLAlchemy==3.1.1 SQLAlchemy==2.0.31 pymysql==1.1.1 flask-cors==5.0.0
```



#### 6. 安装可视化 & 实验工具包



```
pip install matplotlib==3.8.4 seaborn==0.13.2 jupyter==1.0.0
```