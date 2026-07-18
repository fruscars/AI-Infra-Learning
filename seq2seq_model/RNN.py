import torch
import torch.nn.functional as F

# ==========================================
# 1. 文本预处理 (Text Preprocessing)
# ==========================================
raw_text = "machine learning is fun and rnn is powerful"
# 去重构建词表
vocab = sorted(list(set(raw_text)))
vocab_size = len(vocab)
# 建立 字符 <-> 索引 的映射
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}

print(f"词表大小: {vocab_size}, 词表: {''.join(vocab)}")

# ==========================================
# 2. 独热编码 (One-Hot Encoding)
# ==========================================
def to_one_hot(indices, num_classes):
    """将输入的索引列表转换为one-hot张量"""
    # indices 形状: (时间步数/序列长度, )
    # 返回形状: (时间步数, num_classes)
    return F.one_hot(torch.tensor(indices), num_classes=num_classes).float()

# ==========================================
# 3. 循环计算语言模型 (RNN From Scratch)
# ==========================================
class SimpleRNN:
    def __init__(self, vocab_size, num_hiddens):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        
        # 初始化模型参数（输入到隐层、隐层到隐层、隐层到输出）
        # 遵循正态分布初始化，并乘以一个缩放因子
        self.W_xh = torch.randn(vocab_size, num_hiddens) * 0.01
        self.W_hh = torch.randn(num_hiddens, num_hiddens) * 0.01
        self.b_h = torch.zeros(num_hiddens)
        
        self.W_hq = torch.randn(num_hiddens, vocab_size) * 0.01
        self.b_q = torch.zeros(vocab_size)
        
        # 启用梯度
        for param in [self.W_xh, self.W_hh, self.b_h, self.W_hq, self.b_q]:
            param.requires_grad_(True)
            
    def init_state(self):
        """初始化隐状态为全0"""
        return torch.zeros((1, self.num_hiddens))

    def forward(self, inputs, state):
        """
        inputs: 输入序列
          - 期望形状: (时间步数, 1, vocab_size) 
          - 也能兼容: (时间步数, vocab_size)
        state: 初始隐状态
        """
        # 如果输入是二维的 (seq_len, vocab_size)，自动升维成三维 (seq_len, 1, vocab_size)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
            
        outputs = []
        H = state
        # 循环计算的核心：沿着时间步一步一步往前迭代
        for X in inputs:
            # 此时 X 的形状确定为 (1, vocab_size)
            H = torch.tanh(torch.mm(X, self.W_xh) + torch.mm(H, self.W_hh) + self.b_h)
            Y = torch.mm(H, self.W_hq) + self.b_q
            outputs.append(Y)

        return torch.cat(outputs, dim=0), H

# ==========================================
# 4. 预测函数 (Text Prediction)
# ==========================================
def predict(prefix, num_preds, model):
    """
    根据前缀 prefix，预测接下来 num_preds 个字符
    """
    state = model.init_state()
    outputs = [char_to_idx[prefix[0]]]
    
    # 预热阶段：先把前缀喂给 RNN，让它生成对应的隐状态记忆
    for y in prefix[1:]:
        # 将前一个字符转为 one-hot 喂入
        X = to_one_hot([outputs[-1]], model.vocab_size)
        _, state = model.forward(X, state)
        outputs.append(char_to_idx[y])
        
    # 预测阶段：用上一次预测的字符，继续往下预测
    for _ in range(num_preds):
        X = to_one_hot([outputs[-1]], model.vocab_size)
        Y, state = model.forward(X, state)
        # 取概率最大的字符索引作为预测结果
        outputs.append(int(Y.argmax(dim=1).item()))
        
    return ''.join([idx_to_char[i] for i in outputs])

# ==========================================
# 5. 训练模型 (Training Loop)
# ==========================================
# 实例化模型，设置隐状态维度为 32
model = SimpleRNN(vocab_size=vocab_size, num_hiddens=32)

# 准备训练数据
# 输入: "machine learning is fun and rnn is powerfu"
# 目标标签: "achine learning is fun and rnn is powerful" (即输入错开一位后的字符)
input_chars = raw_text[:-1]
target_chars = raw_text[1:]

input_indices = [char_to_idx[c] for c in input_chars]
target_indices = torch.tensor([char_to_idx[c] for c in target_chars])

# 独热编码输入
X_one_hot = to_one_hot(input_indices, vocab_size)

# 开始训练
lr = 0.2
epochs = 1000

print("\n开始训练...")
for epoch in range(epochs):
    state = model.init_state()
    # 前向传播
    y_hat, state = model.forward(X_one_hot, state)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(y_hat, target_indices)
    
    # 反向传播并更新参数
    loss.backward()
    with torch.no_grad():
        for param in [model.W_xh, model.W_hh, model.b_h, model.W_hq, model.b_q]:
            param -= lr * param.grad
            param.grad.zero_()
            
    if (epoch + 1) % 50 == 0:
        # 每次打印用 "machine" 开头预测后续字符的效果
        pred_text = predict("machine", 20, model)
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | 预测效果: '{pred_text}'")

@torch.no_grad()  # 告诉 PyTorch：现在是利用阶段，不需要计算梯度！
def generate_text(prefix, num_preds, model):
    # 1. 初始化隐状态
    state = model.init_state()
    
    # 2. 读入你的“提示词”（Prompt）
    outputs = [char_to_idx[prefix[0]]]
    for y in prefix[1:]:
        X = to_one_hot([outputs[-1]], model.vocab_size)
        _, state = model.forward(X, state)
        outputs.append(char_to_idx[y])
        
    # 3. 开始“自给自足”地往后写
    for _ in range(num_preds):
        # 拿上一步自己预测的字符作为输入
        X = to_one_hot([outputs[-1]], model.vocab_size)
        
        # 前向传播得到下一个字符的概率分布
        Y, state = model.forward(X, state)
        
        # 找到概率最大的那个字符
        next_char_idx = int(Y.argmax(dim=1).item())
        outputs.append(next_char_idx)
        
    # 4. 翻译成人类看得懂的文本
    return ''.join([idx_to_char[i] for i in outputs])

# 体验利用成果：
print(generate_text("machine", 25, model))
# 输出可能是: "machine learning is fun and rnn"