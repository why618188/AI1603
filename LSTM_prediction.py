import torch
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import dgl

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

"""单向LSTM
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        h0 = torch.zeros(1, input.size(0), self.hidden_size)
        c0 = torch.zeros(1, input.size(0), self.hidden_size)
        out, _ = self.lstm.forward(input, (h0, c0))
        out = self.fc.forward(out[:, -1, :])
        return out
"""

# 双向LSTM
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)  # Multiply hidden_size by 2 due to bidirectional LSTM

    def forward(self, input):
        h0 = torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size)  # Multiply num_layers by 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size)
        out, _ = self.lstm.forward(input, (h0, c0))
        out = self.fc.forward(out[:, -1, :])
        return out


model = LSTMModel(24, 128, 24, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)


def model_train():
    loss_history = []
    val_rmse_history = []
    val_epoch = []
    best_val_rmse_loss = 10000
    best_val_epoch = 0
    model.train()
    for epoch in range(500):
        train_logits = model.forward(train_inputs2.unsqueeze(1))
        loss = torch.nn.MSELoss()
        loss1 = loss(train_logits, train_labels)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        if epoch % 10 == 0:
            val_rmse = model_test(model)
            val_rmse_history.append(val_rmse)
            val_epoch.append(epoch)
            if val_rmse < best_val_rmse_loss:
                best_val_rmse_loss = val_rmse
                torch.save(model, './model_train_output/LSTM_best_model_500.pkl')
                best_val_epoch = epoch

        loss_history.append(loss1.item())
        print('Epoch %d | Loss: %.4f' % (epoch, loss1.item()))

    return loss_history, val_epoch, val_rmse_history, best_val_epoch, best_val_rmse_loss

def model_test(model):
    model.eval()
    with torch.no_grad():
        test_logits = model(test_inputs2.unsqueeze(1))
        RMSEloss = round(math.sqrt(mean_squared_error(test_labels.detach().numpy(), test_logits.squeeze().detach().numpy())), 4)
    return RMSEloss

def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.get_cmap('coolwarm')
    normalize = plt.Normalize(vmin=min(loss_history), vmax=max(loss_history))
    colors = [cmap(normalize(y)) for y in loss_history]
    plt.scatter(range(len(loss_history)), loss_history, c=colors, marker='o', s=5)
    plt.grid(True)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('LSTM Training Loss', fontsize=12)
    plt.title('LSTM Training Loss', fontsize=18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    plt.colorbar(sm, label='Loss')

    plt.savefig('./model_train_output/LSTM_training_loss.jpg')
    plt.show()

def plot_rmse_history(loss_history, val_epoch):
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.get_cmap('coolwarm')
    normalize = plt.Normalize(vmin=min(loss_history), vmax=max(loss_history))
    colors = [cmap(normalize(y)) for y in loss_history]
    for i in range(len(val_epoch)-1):
        plt.plot(val_epoch[i:i+2], loss_history[i:i+2], color=colors[i], linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('LSTM Val RMSE loss', fontsize=12)
    plt.title('LSTM Val RMSE Loss', fontsize=18)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    plt.colorbar(sm, label='Loss')

    plt.savefig('./model_train_output/LSTM_val_rmse.jpg')
    plt.show()


loss_his, val_epoch, val_rmse_his, best_loss, best_epoch = model_train()
plot_loss_history(loss_his)
plot_rmse_history(val_rmse_his, val_epoch)
print('val:', best_loss, best_epoch)

model_loaded = torch.load('./model_train_output/LSTM_best_model_500.pkl')
test_logits = model_loaded(test_inputs2.unsqueeze(1))

RMSEloss_sklearn = round(math.sqrt(mean_squared_error(test_labels.detach().numpy(), test_logits.squeeze().detach().numpy())), 4)
MAEloss_sklearn = round(mean_absolute_error(test_labels.detach().numpy(), test_logits.squeeze().detach().numpy()), 4)
print('RMSE:', RMSEloss_sklearn, ' MAE:', MAEloss_sklearn)






