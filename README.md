# Text_classification

This code has been done during an apprenticeship at Atos.


# Classify some texts given a label

This example has been tried on "real" french texts.
The real use case was: 
    - Texts represented problems or improvements that should be done on a software
    - The label is the team which solved the problem
    - + 700 teams were represented and 300 000 texts were annotated.
    - Results of 47.12% accuracy to find the right team to the texts.
    
More explanations on the "presentation" folder.

#### To run: 
    - Launch main.py in Code

Some false data has been done to give you an idea of what is required to launch the code.

## Architecture:
    - Given a file with texts and labels
    - Split the dataset in a training set (80%) and a testing set (20%)
    - In the training set, split the dataset in training set and validation set (20%)
    
    - Train word2vec model on training set
    
    - Embedding of word2vec is used to train a BLSTM network followed by a Softmax layer


In the main.py, you have examples of what I have changed during my experiments and what you can also easily change (number of layers, word2vec replaced by embedding layer, ...)



Improve this code if you want, I'd be glad to accept pull request.
