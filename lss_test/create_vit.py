import torch
import torch.nn as nn
import torch.onnx
from vit_model import vit_large_patch16_224
# import torchsummary

model = vit_large_patch16_224(100)
input_example = torch.randn(1, 3, 224, 224)

# 将模型迁移到GPU（如果可用）以提高性能
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
input_example = input_example.to(device)

print(model)

# # 导出模型到ONNX
# torch.onnx.export(model,  # PyTorch模型
#                   input_example,  # 输入样本
#                   "vit_model.onnx",  # 输出ONNX模型的文件路径
#                   export_params=True,  # 导出模型参数
#                   opset_version=11,  # ONNX的操作集版本
#                   do_constant_folding=True,  # 是否执行常量折叠优化
#                   input_names=["input"],  # 输入节点的名字
#                   output_names=["output"])  # 输出节点的名字

# print("Model exported successfully.")