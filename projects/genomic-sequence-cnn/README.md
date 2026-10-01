For my models, I decided to use two convolutional layers followed by a linear layer. For each convolutional layer, I batch normalized the data before applying ReLu activation. For the linear layer, I used ReLu activation as well. 

I found that more than two convolutional layers led to overfitting. 

The loss function I used was `torch.nn.BCEWithLogitsLoss()`. I used it because it supposedly more numerically stable. 

I used Optuna to optimize the following hyperparameters:
- batch size
- kernel size
- number of filters
- type of optimizer (adam or sgd)
- learning rate

I tuned the hyperparameters with the training data and then evaluated on the validation data. 
Then, I retrained the model on both the training data and validation data and evaluated the test set on this trained model. 