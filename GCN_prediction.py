# 使用构建好的数据图谱进行车流预测
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import math
import dgl
import torch
import matplotlib.pyplot as plt

g = dgl.load_graphs('shanghai_graph.bin')[0][0]
nodes_list = g.nodes().numpy()
g_train = dgl.node_subgraph(g, nodes_list[:int((3 / 5) * len(nodes_list))])
g_train = dgl.add_self_loop(g_train)
g_val = dgl.node_subgraph(g, nodes_list[int((3 / 5) * len(nodes_list)):int((4 / 5) * len(nodes_list))])
g_val = dgl.add_self_loop(g_val)
g_test = dgl.node_subgraph(g, nodes_list[int((4 / 5) * len(nodes_list)):])
g_test = dgl.add_self_loop(g_test)

train_inputs2 = g_train.ndata['x_count'].float()
train_labels = g_train.ndata['y'].float()
val_inputs2 = g_val.ndata['x_count'].float()
val_labels = g_val.ndata['y'].float()
test_inputs2 = g_test.ndata['x_count'].float()
test_labels = g_test.ndata['y'].float()


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size1)
        self.conv2 = GraphConv(hidden_size1, hidden_size2)
        self.conv3 = GraphConv(hidden_size2, hidden_size3)
        self.fc = torch.nn.Linear(hidden_size3, num_classes)
        self.activate = torch.nn.Tanh()

    def forward(self, g, inputs2):
        h = self.conv1(g, inputs2)
        h = self.activate(h)
        h = self.conv2(g, h)
        h = self.activate(h)
        h = self.conv3(g, h)
        h = self.activate(h)
        h = self.fc(h)
        return h


net1 = GCN(24, 128, 64, 24, 24)
optimizer = torch.optim.Adam(net1.parameters(), lr=0.01)


def model_train():
    loss_history = []
    val_rmse_history = []
    val_epoch = []
    best_val_rmse_loss = 10000
    best_val_epoch = 0
    net1.train()  # 训练模式
    for epoch in range(500):  # 完整遍历一遍训练集 一个epoch做一次更新
        train_logits = net1(g_train, train_inputs2)  # 所有数据前向传播 （N,7）
        loss = nn.MSELoss()
        loss1 = loss(train_logits, train_labels)
        optimizer.zero_grad()  # 清空梯度
        loss1.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新
        if epoch % 10 == 0:
            val_rmse = model_test(net1)  # 计算当前模型在验证集上的准确率
            val_rmse_history.append(val_rmse)
            val_epoch.append(epoch)
            if val_rmse < best_val_rmse_loss:
                best_val_rmse_loss = val_rmse
                torch.save(net1, './model_train_output/GCN_best_model_500.pkl')
                best_val_epoch = epoch
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss1.item())
        print('Epoch %d | Loss: %.4f' % (epoch, loss1.item()))

    return loss_history, val_epoch, val_rmse_history, best_val_epoch, best_val_rmse_loss


def model_test(model):
    model.eval()
    with torch.no_grad():
        logits = net1(g_val, val_inputs2)
        RMSEloss = round(math.sqrt(mean_squared_error(val_labels.detach().numpy(), logits.detach().numpy())), 4)
    return RMSEloss


def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.get_cmap('coolwarm')
    normalize = plt.Normalize(vmin=min(loss_history), vmax=max(loss_history))
    colors = [cmap(normalize(y)) for y in loss_history]
    plt.scatter(range(len(loss_history)), loss_history, c=colors, marker='o', s=5)
    plt.grid(True)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('GCN Training Loss', fontsize=12)
    plt.title('GCN Training Loss', fontsize=18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    plt.colorbar(sm, label='Loss')

    plt.savefig('./model_train_output/GCN_training_loss.jpg')
    plt.show()

def plot_rmse_history(loss_history, val_epoch):
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.get_cmap('coolwarm')
    normalize = plt.Normalize(vmin=min(loss_history), vmax=max(loss_history))
    colors = [cmap(normalize(y)) for y in loss_history]
    for i in range(len(val_epoch)-1):
        plt.plot(val_epoch[i:i+2], loss_history[i:i+2], color=colors[i], linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('GCN Val RMSE loss', fontsize=12)
    plt.title('GCN Val RMSE Loss', fontsize=18)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    plt.colorbar(sm, label='Loss')

    plt.savefig('./model_train_output/GCN_val_rmse.jpg')
    plt.show()


loss_his, val_epoch, val_rmse_his, best_loss, best_epoch = model_train()
plot_loss_history(loss_his)
plot_rmse_history(val_rmse_his, val_epoch)
print('val:', best_loss, best_epoch)

net_loaded = torch.load('./model_train_output/GCN_best_model_500.pkl')
test_logits = net_loaded(g_test, test_inputs2)

RMSEloss_sklearn = round(math.sqrt(mean_squared_error(test_labels.detach().numpy(), test_logits.detach().numpy())), 4)
MAEloss_sklearn = round(mean_absolute_error(test_labels.detach().numpy(), test_logits.detach().numpy()), 4)
print('RMSE:', RMSEloss_sklearn, ' MAE:', MAEloss_sklearn)
