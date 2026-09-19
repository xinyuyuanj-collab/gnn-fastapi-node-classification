from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# ==========================================
# 1. 服务初始化与配置
# ==========================================
app = FastAPI(
    title="Graph Representation & Node Classification API",
    description="基于图卷积神经网络 (GCN) 与 Cora 数据集的文献节点分类与向量表征推理微服务",
    version="1.0.0"
)


# ==========================================
# 2. 图神经网络模型定义 (GCN Encoder)
# ==========================================
class GCNEncoder(nn.Module):
    """
    两层图卷积神经网络编码器 (GCN)
    用于提取节点拓扑结构特征并输出分类概率分布
    """
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int):
        super(GCNEncoder, self).__init__()
        # 第一层图卷积：将节点输入特征映射到隐藏表征空间
        self.conv1 = GCNConv(in_channels, hidden_dim)
        # 第二层图卷积：将隐藏特征映射到类别分布空间
        self.conv2 = GCNConv(hidden_dim, out_channels)

    def forward(self, x, edge_index):
        # 隐藏层特征提取与非线性激活
        h = self.conv1(x, edge_index)
        h_relu = F.relu(h)
        # 节点分类输出层
        out = self.conv2(h_relu, edge_index)
        return out, h  # 返回分类 logits 与 512 维嵌入向量 (Embedding)


# ==========================================
# 3. 运行时数据加载与模型预热
# ==========================================
print("[INFO] 正在初始化图数据集与模型推理环境...")

# 加载 Cora 图数据集 (如不存在则自动从镜像源获取)
dataset = Planetoid(root="./data", name="Cora")
graph_data = dataset[0]

# 实例化模型拓扑结构：输入维度 1433，隐藏层维度 512，分类数 7
model = GCNEncoder(
    in_channels=dataset.num_features,
    hidden_dim=512,
    out_channels=dataset.num_classes
)

# 可选：加载预训练权重
# if os.path.exists("checkpoint.pth"):
#     model.load_state_dict(torch.load("checkpoint.pth", map_location=torch.device('cpu')))

# 切换为评估推理模式
model.eval()
print("[INFO] 模型就绪，推理服务启动完成。")


# ==========================================
# 4. 数据传输对象模式 (DTO Schemas)
# ==========================================
class NodeQuery(BaseModel):
    node_id: int = Field(..., ge=0, description="图拓扑节点索引编号 (Node Index)")


# ==========================================
# 5. 推理接口：节点类别预测 (POST /predict)
# ==========================================
@app.post("/predict", summary="节点分类预测接口")
def predict_node_class(query: NodeQuery):
    # 边界合法性校验
    if query.node_id >= graph_data.num_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"节点索引越界: 输入 ID 为 {query.node_id}，合法范围为 [0, {graph_data.num_nodes - 1}]。"
        )

    # 模型推理 (关闭梯度计算以提高吞吐量)
    with torch.no_grad():
        out, _ = model(graph_data.x, graph_data.edge_index)
        predicted_label = out[query.node_id].argmax().item()

    return {
        "status": "success",
        "node_id": query.node_id,
        "predicted_class": predicted_label,
        "dataset": "Cora"
    }


# ==========================================
# 6. 特征接口：节点嵌入向量提取 (POST /embed)
# ==========================================
@app.post("/embed", summary="节点高维表征向量提取接口")
def get_node_embedding(query: NodeQuery):
    # 边界合法性校验
    if query.node_id >= graph_data.num_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"节点索引越界: 输入 ID 为 {query.node_id}，合法范围为 [0, {graph_data.num_nodes - 1}]。"
        )

    with torch.no_grad():
        _, embeddings = model(graph_data.x, graph_data.edge_index)
        node_vec = embeddings[query.node_id].tolist()

    return {
        "status": "success",
        "node_id": query.node_id,
        "vector_dimension": len(node_vec),
        "embedding_vector": node_vec
    }
