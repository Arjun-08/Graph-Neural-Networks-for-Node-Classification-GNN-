# Graph Neural Networks for Node Classification

Implementation of a **Graph Convolutional Network (GCN)** for node classification using **PyTorch Geometric** and the **Cora citation network**.

The project demonstrates how Graph Neural Networks incorporate both node features and graph structure to learn node representations and classify nodes in a transductive learning setting.


In node classification, only a subset of nodes has known ground-truth labels. The objective is to learn from these labeled nodes and predict the classes of the remaining nodes.

This project uses the **Cora citation network**, where:

* Each node represents a document.
* Each document is represented by a 1433-dimensional bag-of-words feature vector.
* Edges represent citation relationships between documents.
* The classification task contains 7 document classes.

The implementation uses `GCNConv` layers from PyTorch Geometric to perform graph-based message passing and representation learning.

## Dataset

### Cora Citation Network

The Cora dataset is part of the **Planetoid benchmark suite** and is loaded using PyTorch Geometric's `Planetoid` dataset interface.

Feature normalization is applied using `NormalizeFeatures()`:

```python
dataset = Planetoid(
    root='data/Planetoid',
    name='Cora',
    transform=NormalizeFeatures()
)
```

The dataset contains:

| Property            |  Value |
| ------------------- | -----: |
| Graphs              |      1 |
| Nodes               |  2,708 |
| Node features       |  1,433 |
| Classes             |      7 |
| Edges               | 10,556 |
| Training nodes      |    140 |
| Training label rate | ~5.17% |
| Isolated nodes      |     No |
| Self-loops          |     No |
| Undirected graph    |    Yes |

The graph statistics are obtained directly from the loaded PyTorch Geometric dataset.

### Graph Structure

The Cora dataset can be viewed as:

```text
                 Citation Network

              ┌───────────┐
              │ Document  │
              │   Node    │
              └─────┬─────┘
                    │
          ┌─────────┼─────────┐
          │         │         │
          v         v         v
      Document   Document   Document
        Node       Node       Node
          │
          └────── Citation ──────>
```

Each node contains its own feature vector, while the edges provide information about relationships between nodes.

## Model Architecture

A two-layer Graph Convolutional Network is implemented.

```text
Input Features
1433 dimensions
       |
       v
   GCNConv
1433 → 20
       |
       v
     ReLU
       |
       v
   GCNConv
 20 → 7
       |
       v
Class Logits
```

The network uses:

* Input dimension: 1433
* Hidden dimension: 20
* Output dimension: 7
* Two graph convolution layers
* ReLU activation between the layers
* Random seed: 42

The model uses the graph's `edge_index` during both graph convolution operations, allowing information from neighboring nodes to contribute to each node's representation.

## Graph Convolution

A graph convolution differs from a conventional fully connected layer because it aggregates information from neighboring nodes.

The GCN operation used in this project is:

<img width="491" height="121" alt="image" src="https://github.com/user-attachments/assets/6f7c6bd3-cf32-42ac-80e6-a7700858768e" />


where:

* $\mathbf{x}_v^{(\ell)}$ is the representation of node $v$ at layer $\ell$.
* $\mathcal{N}(v)$ represents the neighbors of node $v$.
* $\mathbf{W}^{(\ell+1)}$ is a trainable weight matrix.
* $c_{w,v}$ is the normalization coefficient for the edge.

Unlike a standard linear layer,

<img width="275" height="70" alt="image" src="https://github.com/user-attachments/assets/dfd73f02-9a73-4f24-882d-20ba2706a74a" />


the GCN incorporates neighboring node information during representation learning.

## Implementation Workflow
The script performs the following steps:

1. Loads the Cora citation network.
2. Normalizes the node features.
3. Displays dataset and graph statistics.
4. Initializes the two-layer GCN.
5. Trains the model for 100 epochs.
6. Reports training loss and accuracy.
7. Evaluates the model on the test nodes.
8. Generates a t-SNE visualization of the learned embeddings.## Implementation Workflow

## Training Configuration

The model is trained using the following configuration:

| Hyperparameter  |              Value |
| --------------- | -----------------: |
| Architecture    |        2-layer GCN |
| Hidden channels |                 20 |
| Optimizer       |               Adam |
| Learning rate   |               0.01 |
| Weight decay    |           5 × 10⁻⁴ |
| Loss function   | Cross-Entropy Loss |
| Training epochs |                100 |
| Random seed     |                 42 |

The training loss is calculated only over nodes selected by the training mask:

```python
loss = criterion(
    out[data.train_mask],
    data.y[data.train_mask]
)
```

Training accuracy is calculated using the same training mask. The model is trained for **100 epochs**, with the loss and training accuracy reported after every epoch.



## Test Results

After training, the model is evaluated using the test mask.

The final test accuracy is calculated as:

```python
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
```

The final results reported :

| Training epochs         | 100    |
| ----------------------- | ------ |
| Final training loss     | 0.3368 |
| Final training accuracy | 99.29% |
| Test accuracy           | 81.50% |


The test evaluation uses only the nodes selected by `data.test_mask`.

## Embedding Visualization

The project also visualizes the representations learned by the GCN.

After training, the model output is passed to **t-SNE** to reduce the learned representations to two dimensions. The visualization function performs the dimensionality reduction and plots the resulting representations using the ground-truth class labels.


<img width="794" height="790" alt="image" src="https://github.com/user-attachments/assets/610ae5c8-f22a-4276-a123-1d8eebcfc32b" />

This visualization provides a qualitative view of how well the learned representations separate the different node classes.




## References

* Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.*
* Yang, Z., Cohen, W. W., & Salakhutdinov, R. (2016). *Revisiting Semi-Supervised Learning with Graph Embeddings.*
* PyTorch Geometric documentation.

## Acknowledgements

This project uses the Cora dataset from the Planetoid benchmark and the PyTorch Geometric framework for implementing graph neural network operations.

