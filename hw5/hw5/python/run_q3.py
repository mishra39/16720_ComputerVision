import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import copy

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 80
learning_rate = 4.5e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1], hidden_size , params,'layer1')
initialize_weights(hidden_size , train_y.shape[1],params,'output')

W_orig_layer1 = copy.deepcopy(params['W'+'layer1'])
# with default settings, you should get loss < 150 and accuracy > 80%
train_acc = []
train_loss = []
valid_accuracy = []
valid_loss = []

for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        # Implement backwards!
        backwards(delta2,params,'layer1',sigmoid_deriv)
        # apply gradient
        params['W'+'layer1'] -= learning_rate * params['grad_W' + 'layer1']
        params['b'+'layer1'] -= learning_rate * params['grad_b' + 'layer1']
        params['W'+'output'] -= learning_rate * params['grad_W' + 'output']
        params['b'+'output'] -= learning_rate * params['grad_b' + 'output']

    total_acc = total_acc / batch_num
    train_acc.append(total_acc)
    train_loss.append(total_loss)

    # run on validation set and report accuracy! should be above 75%
    valid_acc = None
    h1 = forward(valid_x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    valid_loss.append(loss)
    valid_accuracy.append(valid_acc)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        print('Validation accuracy: ',valid_acc)


# 3.1 Plotting
plt.figure("Accuracy")
plt.plot(range(max_iters), train_acc)
plt.plot(range(max_iters), valid_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'])
plt.show()

plt.figure("Cross-Entropy Loss")
plt.plot(range(max_iters), train_loss)
plt.plot(range(max_iters), valid_loss)
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend(['Training', 'Validation'])
plt.show()

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

with open('q3_weights.pickle', 'rb') as handle:
    saved_params = pickle.load(handle)

w_layer1 = saved_params['W'+'layer1']
# visualize weights here
fig = plt.figure("Learned Weight Image Grid")
grid = ImageGrid(fig, 111, (8,8))

for i in range(hidden_size):
    w_learn = w_layer1[:,i].reshape(32,32)
    grid[i].imshow(w_learn)
    plt.axis('off')
plt.show()

fig = plt.figure("Original Weight Image Grid")
grid = ImageGrid(fig, 111, (8,8))

for i in range(hidden_size):
    w_orig = W_orig_layer1[:,i].reshape(32,32)
    grid[i].imshow(w_orig)
    plt.axis('off')
plt.show()

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
h1 = forward(test_x,saved_params,'layer1')
probs = forward(h1,saved_params,'output',softmax)
pred = np.argmax(probs, axis=1)
act  = np.argmax(test_y, axis=1)

for i in range(act.shape[0]):
    confusion_matrix[act[i],pred[i]] += 1



import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()