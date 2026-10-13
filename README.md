

The repository is organized as follows:

```
MSTHMN/
│
├── Data/
│   └── PEMS04/
│       ├── adj_mx.pkl      # Distance matrix file
│       └── data.dat      # Traffic data file
│
├── layers.py               # Defines custom model layers 
├── model.py                # Defines the main MSTHMN model architecture
├── utils.py                # Contains utility functions for data loading, processing, and evaluation
├── train.py                # Main script for training and evaluating the model
└── README.md               # This file
```

## Requirements

To run this project, you need the following libraries installed. Using a virtual environment (e.g., conda) is highly recommended.

* Python 3.10.13
* PyTorch 2.5.1
* NumPy 2.2.6
* SciPy 1.15.2
* scikit-learn 1.7.2


```bash
python train.py --device 0 --dataset_name PEMS04
```

You can customize the training process with various arguments:

* `--epochs`: Number of training epochs (default: 100)
* `--batch_size`: Batch size for training (default: 32)
* `--lr`: Learning rate (default: 5e-4)
* `--k`: Number of neighbors for hypergraph construction (default: 8)

For a full list of options, please refer to the `train.py` script.

## Results

After training, the model's performance on the test set will be printed to the console. The evaluation metrics include Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Squared Error (RMSE).
