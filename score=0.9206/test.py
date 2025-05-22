import torch

# 加载模型
model_path = 'best0106.pkl'  # 更改为你的模型文件路径
model = torch.load(model_path, map_location=torch.device('cpu'))

# 如果加载的是模型的状态字典，则需要先创建模型实例，并加载状态字典
# model = YourModelClass()  # 你的模型类
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
import matplotlib.pyplot as plt

# 遍历模型的权重参数
for name, param in model.named_parameters():
    if 'weight' in name:  # 仅针对权重参数，跳过偏置等其他参数
        plt.figure(figsize=(10, 7))
        # 将权重数据转换为一维数组
        weight_data = param.data.numpy().flatten()
        plt.hist(weight_data, bins=50, alpha=0.7)
        plt.title(f'Weight Distribution of Layer: {name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.show()
