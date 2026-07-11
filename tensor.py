#基于pytorch nn库的线性函数拟合

import torch
import torch.nn as nn


#构建人工数据集
x = torch.randn(100, 1)
y = 2 * x + 1 + torch.randn(100, 1) * 0.1#添加噪声)

#定义线性回归模型:
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度为1，输出维度为1

    def forward(self, x):#
        return self.linear(x)
    
#训练模型
model = LinearRegression()
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器优化器

#训练循环()
for epoch in range(1000):

    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')     

#返回模型参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
