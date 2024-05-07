import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F

# # 定义一个包含linear的模型
# class LinearModel(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(LinearModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 1, 3, 2)
#         self.linear = nn.Linear(in_features, out_features)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = torch.reshape(x, [1, -1])
#         return self.linear(x)
# model = LinearModel(12321, 32)  


# 定义一个包含Embedding的模型
# class EmbeddingModel(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(EmbeddingModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 1, 3, 2)
#         self.linear = nn.Linear(in_features, out_features)
#         self.embedding = nn.Embedding(10, out_features)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = torch.reshape(x, [1, -1])
#         x = self.linear(x)
#         y = torch.LongTensor([1]).to('cuda:0')
#         y = self.embedding(y)
#         return x * y
# model = EmbeddingModel(12321, 32)


# # 定义一个包含RMSNorm的模型
# class RMSNormModel(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(RMSNormModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 1, 3, 2)
#         self.linear = nn.Linear(in_features, out_features)
#         self.eps = 1e-6

#     def forward(self, x):
#         x = self.conv1(x)
#         x = torch.reshape(x, [1, -1])
#         x = self.linear(x)
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
# model = RMSNormModel(12321, 32) 


# 定义一个包含RMSNorm的模型
class TestModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 3, 2)
        self.conv2 = nn.Conv2d(1, 1, 3, 2)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = x.pow(2)
        x = self.conv2(x)
        x = torch.reshape(x, [1, -1])
        x = self.linear(x)
        return x
model = TestModel(3025, 32)

# 定义一个包含mlp的模型
# class MlpModel(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(TestModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 1, 3, 2)
#         self.linear = nn.Linear(in_features, out_features)
#         self.linear2 = nn.Linear(out_features, out_features // 2)
#         self.linear3 = nn.Linear(out_features, out_features // 2)
#         self.linear4 = nn.Linear(out_features // 2, out_features)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = torch.reshape(x, [1, -1])
#         x = self.linear(x)
#         x1 = self.linear2(x)
#         x2 = self.linear3(x)
#         x3 = x1 + F.silu(x2)
# #         return self.linear4(x3)
# model = TestModel(12321, 32)  


input_example = torch.randn(1, 3, 224, 224)  # 假设输入样本的形状为 (batch_size=1, in_features=10)

# 将模型迁移到GPU（如果可用）以提高性能
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
input_example = input_example.to(device)
out = model(input_example)

# 设置模型为评估模式
model.eval()

# 导出模型到ONNX
torch.onnx.export(model,  # PyTorch模型
                  input_example,  # 输入样本
                  "test_model.onnx",  # 输出ONNX模型的文件路径
                  export_params=True,  # 导出模型参数
                  opset_version=11,  # ONNX的操作集版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=["input"],  # 输入节点的名字
                  output_names=["output"])  # 输出节点的名字

print("Model exported successfully.")