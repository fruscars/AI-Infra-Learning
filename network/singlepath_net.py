#假设是三层神经网络
import numpy as np
import matplotlib.pyplot as plt
#人工数据集

#神经网络层（对节点分类，即统一又分类）
# ---------- 线性层节点 ----------
class Linear:
    def __init__(self, in_features, out_features):
        # 参数初始化（学习需要现在要使用sigmoid和Xavier初始化），w为权重，b为偏置
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))
        # 占位符，反向传播时填充
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
       #x为层输入，out为层输出
        self.x = x.copy()          # ★ 存下输入，反向时要用
        self.out = x @ self.W + self.b   
        return self.out

    def backward(self, grad_output):
        """
        grad_output: ∂L/∂out，上游传来的损失对本层输出的梯度
        本方法拆成两步：
        第一步：写出当前层运算对各个输入的局部偏导（仅由 out = x@W + b 决定）
        第二步：利用链式法则，将局部偏导与上游梯度 grad_output 结合，
                得到损失函数对本层参数和输入的偏导数。
        """
        # ========== 第一步：局部偏导（只与本层运算有关） ==========
        # 对于 out = x @ W + b：
        # ∂out/∂W 的结构是 x 的转置（具体作用在 grad_output 上时会变成 x.T @ grad_output）
        # ∂out/∂b = 1（广播，反向时等价于对 grad_output 沿 batch 求和）
        # ∂out/∂x = W 的转置（作用在 grad_output 上会变成 grad_output @ W.T）

        # ========== 第二步：链式法则 × grad_output = 损失关于各变量的梯度 ==========
        # ∂L/∂W = ∂L/∂out · ∂out/∂W  →  x.T @ grad_output
        self.grad_W = self.x.T @ grad_output

        # ∂L/∂b = ∂L/∂out · ∂out/∂b  →  sum(grad_output, axis=0) （因为 b 对每个样本的导数都是 1）
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)

        # ∂L/∂x = ∂L/∂out · ∂out/∂x  →  grad_output @ W.T
        grad_input = grad_output @ self.W.T

        # 返回给前一层的梯度（∂L/∂x），让链式法则继续传递,这个损失函数对输入的导数放到反向最后一层计算不更好吗？输入值与损失函数的偏导能复用参数与损失函数偏导的值吗？？
        return grad_input

    def update(self, lr):
        # 使用损失函数关于权重和偏置的偏导数（已经存在 self.grad_W 和 self.grad_b 中）
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b


# ---------- sigmoid激活函数 ----------
class Sigmoid:  
    #输入与输出
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
   
    def backward(self, grad_output):
        # 计算sigmoid的梯度
        grad_input = grad_output * self.out * (1 - self.out)
        return grad_input

#---------- 构建激活函数三层神经网络 ----------
class ThreeLayerNet:
    def __init__(self, in_dim, h1_dim, h2_dim, out_dim):
        # 第一层：线性 + sigmoid
        self.fc1 = Linear(in_dim, h1_dim)
        self.sigmoid1 = Sigmoid()
        # 第二层：线性 + sigmoid
        self.fc2 = Linear(h1_dim, h2_dim)
        self.sigmoid2 = Sigmoid()
        # 第三层：线性（输出层，不加激活，用于回归）
        self.fc3 = Linear(h2_dim, out_dim)

    def forward(self, x):
        # 前向传播：按顺序让数据流过每一个节点
        out = self.fc1.forward(x)      # 线性节点1
        out = self.sigmoid1.forward(out)  # 激活节点1
        out = self.fc2.forward(out)    # 线性节点2
        out = self.sigmoid2.forward(out)  # 激活节点2
        out = self.fc3.forward(out)    # 输出线性节点
        return out

    def backward(self, grad_output):
        # 反向传播：逆序调用各节点的 backward，梯度逐层向前传递
        grad = self.fc3.backward(grad_output)    # 输出层
        grad = self.sigmoid2.backward(grad)      # 第二层激活
        grad = self.fc2.backward(grad)           # 第二层线性
        grad = self.sigmoid1.backward(grad)      # 第一层激活
        grad = self.fc1.backward(grad)           # 第一层线性（grad 不再使用）
        return grad  # 如果需要继续往前传，可以返回；这里作为最前层也可以忽略

    def update(self, lr):
        # 更新所有包含参数的节点（只有 Linear 有参数）
        self.fc1.update(lr)
        self.fc2.update(lr)
        self.fc3.update(lr)

# ==================== 损失函数（MSE） ====================
def mse_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    diff = y_pred - y_true
    loss = np.mean(diff ** 2)
    grad = 2 * diff / batch_size
    return loss, grad




# ==================== 数据生成 ====================
#sin函数拟合
np.random.seed(42)
X = np.random.uniform(-np.pi, np.pi, 200).reshape(-1, 1)
y = np.sin(X)

# ==================== 训练 ====================
model = ThreeLayerNet(in_dim=1, h1_dim=16, h2_dim=16, out_dim=1)
lr = 0.1
epochs = 5000
loss_history = []

for epoch in range(epochs):
    y_pred = model.forward(X)
    loss, dL_dy = mse_loss(y_pred, y)
    model.backward(dL_dy)
    model.update(lr)
    loss_history.append(loss)
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}, Loss: {loss:.6f}")

# ==================== 可视化 ====================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('Training Loss')

plt.subplot(1,2,2)
X_test = np.linspace(-np.pi, np.pi, 300).reshape(-1,1)
y_pred_test = model.forward(X_test)
plt.scatter(X, y, s=10, label='Train data')
plt.plot(X_test, np.sin(X_test), 'r-', label='True sin(x)')
plt.plot(X_test, y_pred_test, 'g--', label='Model prediction')
plt.legend()
plt.title('Fitting sin(x)')
plt.show()
