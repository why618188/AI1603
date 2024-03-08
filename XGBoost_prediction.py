from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
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

class XGBoost(torch.nn.Module):
    def __init__(self):
        super(XGBoost, self).__init__()
        self.model = None

    def forward(self, inputs):
        return self.model.predict(xgb.DMatrix(inputs))

model = XGBoost()
xgb_params = {
    'objective': 'reg:squarederror',
    'eta': 0.01,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_matric': 'rmse'
}

def model_train():
    dtrain = xgb.DMatrix(train_inputs2.numpy(), label=train_labels.numpy())
    dval = xgb.DMatrix(val_inputs2.numpy(), label=val_labels.numpy())
    watchList = [(dtrain, 'train'), (dval, 'val')]
    evalsResult = {}
    model.model = xgb.train(xgb_params, dtrain, 500, watchList, evals_result=evalsResult, early_stopping_rounds=20)
    return evalsResult['val']['rmse']

def model_test(model):
    dtest = xgb.DMatrix(test_inputs2.numpy())
    return model.model.predict(dtest)

def plot_rmse_loss_history(loss_history):
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.get_cmap('coolwarm')
    normalize = plt.Normalize(vmin=min(loss_history), vmax=max(loss_history))
    colors = [cmap(normalize(y)) for y in loss_history]
    plt.scatter(range(len(loss_history)), loss_history, c=colors, marker='o', s=5)
    plt.grid(True)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('XGBoost Training RMSE Loss', fontsize=12)
    plt.title('XGBoost Training RMSE Loss', fontsize=18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    plt.colorbar(sm, label='Loss')

    plt.savefig('./model_train_output/XGB_training_loss.jpg')
    plt.show()

loss_his = model_train()
plot_rmse_loss_history(loss_his)

test_logits = model_test(model)

RMSEloss_sklearn = round(math.sqrt(mean_squared_error(test_labels.detach().numpy(), test_logits)), 4)
MAEloss_sklearn = round(mean_absolute_error(test_labels.detach().numpy(), test_logits), 4)
print('RMSE:', RMSEloss_sklearn, ' MAE:', MAEloss_sklearn)