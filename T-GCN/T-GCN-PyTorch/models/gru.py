import argparse
import torch
import torch.nn as nn


# 自定义的线性层，用于GRU中的线性变换，连接输入和隐藏状态。
class GRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(GRULinear, self).__init__()
        self._num_gru_units = num_gru_units  # GRU的隐层单元数量
        self._output_dim = output_dim
        self._bias_init_value = bias  # 偏置的初始值，默认为0.0

        # 定义权重矩阵参数，尺寸为(num_gru_units + 1, output_dim)
        # 其中多加的一维用于包括输入值
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))  # 偏置向量
        self.reset_parameters()

    # 初始化权重矩阵和偏置向量，以提高模型的训练效果和稳定性。
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape

        # inputs (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))

        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )

        # 关键一步：拼接输入和隐藏状态
        # [inputs, hidden_state] "[x, h]" (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)

        # [x, h] (batch_size * num_nodes, gru_units + 1)，便于下一步与权重矩阵 self.weights 相乘
        concatenation = concatenation.reshape((-1, self._num_gru_units + 1))

        # [x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = concatenation @ self.weights + self.biases
        # [x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # [x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    # 通常用于模型配置、调试或日志记录，可帮助了解当前类实例的设置以便进行测试或调整。
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


# 定义GRU单元的核心计算，包括计算重置门r和更新门u，然后用它们更新隐藏状态。
class GRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        # 用于计算门控的线性层
        # 输出维度为2 * hidden_dim，因为通过计算同时可以得到更新门、重置门
        self.linear1 = GRULinear(self._hidden_dim, self._hidden_dim * 2, bias=1.0)

        # 用于计算候选隐藏状态的线性层
        # 输出维度为单一的hidden_dim，因为通过计算只得到候选隐藏状态
        self.linear2 = GRULinear(self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid([x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.linear1(inputs, hidden_state))  # 将 inputs 和 hidden_state 组合进行线性变换

        # r (batch_size, num_nodes * num_gru_units)
        # u (batch_size, num_nodes * num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)  # 拆分线性变换结果，将其分成两个部分分别作为重置门 r 和更新门 u

        # c = tanh([x, (r * h)]W + b)
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.linear2(inputs, r * hidden_state))  # 计算候选隐藏状态 c

        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1 - u) * c  # 计算新的隐藏状态 h
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


# 定义完整的多步时间序列模型，将输入序列中的每个时间步通过GRUCell单步更新依次处理，最终输出序列的最后一个时间步的结果。
class GRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        super(GRU, self).__init__()
        self._input_dim = input_dim  # num_nodes for prediction
        self._hidden_dim = hidden_dim
        self.gru_cell = GRUCell(self._input_dim, self._hidden_dim)   # 使用GRUCell来创建GRU模型

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape

        # 检查输入节点数量是否符合GRU的预期输入维度
        assert self._input_dim == num_nodes  # assert用来测试表示式，其返回值为假，就会触发异常。

        # 初始化输出列表和隐藏状态
        outputs = list()
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )

        # 遍历每个时间步
        for i in range(seq_len):
            output, hidden_state = self.gru_cell(inputs[:, i, :], hidden_state)  # 使用GRU单元更新隐藏状态
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs.append(output)  # 保存每个时间步的输出
        last_output = outputs[-1]  # 获取最后一个时间步的输出，作为整个GRU的输出结果
        return last_output

    # 为GRU模型添加超参数
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)  # 添加一个新的参数，指定GRU的隐藏层维度
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
