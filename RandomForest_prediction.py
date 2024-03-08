import math
import dgl
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

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

class RandomForest(torch.nn.Module):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.model = None

    def forward(self, inputs):
        return self.model.predict(inputs)

model = RandomForest()

def model_train():
    model.model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.model.fit(train_inputs2.numpy(), train_labels.numpy())

def model_test(model):
    return model.model.predict(test_inputs2.numpy())

model_train()
test_logits = model_test(model)

RMSEloss_sklearn = round(math.sqrt(mean_squared_error(test_labels.detach().numpy(), test_logits)), 4)
MAEloss_sklearn = round(mean_absolute_error(test_labels.detach().numpy(), test_logits), 4)
print('RMSE:', RMSEloss_sklearn, ' MAE:', MAEloss_sklearn)
