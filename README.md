# 🧠 基于 GCN 图神经网络与 FastAPI 的学术文献节点分类微服务

本项目基于 **PyTorch Geometric (PyG)** 构建了两层图卷积网络 (GCN)，在经典的学术论文引用网络 **Cora 数据集** 上实现了节点特征编码与多类别分类推理，并使用 **FastAPI** 封装为高性能异步 RESTful API 微服务。

---

## 🌟 核心技术与特性
- 📐 **图深度学习框架**：采用 PyTorch + PyTorch Geometric 构建图神经网络模型
- 📊 **引用网络建模**：针对 Cora 数据集（1433 维词袋特征，7 类论文类别）进行图拓扑结构处理
- ⚡ **微服务封装**：基于 ASGI 架构的 FastAPI，提供低延迟的节点推理与表征提取接口
- 🔍 **数据自动化**：内置图数据集的自动获取与预处理管线

---

## 🛠 本地快速启动指南

### 1. 安装核心依赖
\`\`\`bash
pip install fastapi uvicorn torch torchvision torchaudio
pip install torch_geometric
\`\`\`

### 2. 启动 API 服务
\`\`\`bash
uvicorn main:app --reload
\`\`\`

### 3. 访问交互式接口文档
启动后，在浏览器访问：
👉 `http://127.0.0.1:8000/docs`
