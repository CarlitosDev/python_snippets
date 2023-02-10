'''

https://github.com/microsoft/petridishnn

From:
https://towardsdatascience.com/microsoft-introduces-project-petridish-to-find-the-best-neural-network-for-your-problem-f3022bee5fce

Neural architecture search(NAS) is one of the hottest trends in modern deep learning technologies.
NAS methods focus on finding a suitable neural network architecture for a given problem and dataset.



There is another problem in machine learning that resembles the challenges of NAS techniques: feature selection.




Two Types of NAS: Forward Search vs. Backward Search


Petridish is a forward-search NAS method inspired by feature selection and gradient boosting techniques. 
The algorithm works by creating a gallery of models to choose from as its search output and then 
incorporating stop-forward and stop-gradient layers to more efficiently identify beneficial candidates for building that gallery,
and uses asynchronous training.

Three fundamental phases:

PHASE 0: Petridish starts with some parent model, a very small human-written model with one or two layers or a model already found by domain experts on a dataset.

PHASE 1: Petridish connects the candidate layers to the parent model using stop-gradient and 
stop-forward layers and partially train it. The candidate layers can be any bag of operations in the search space. 

Using stop-gradient and stop-forward layers allows gradients with respect to the candidates 
to be accumulated without affecting the model’s forward activations and backward gradients. Without the
 stop-gradient and stop-forward layers, it would be difficult to determine which candidate layers are 
contributing what to the parent model’s performance and would require separate training 
if you wanted to see their respective contributions, increasing costs.

PHASE 2: If a particular candidate or set of candidates is found to be beneficial to the model, 
then we remove the stop-gradient and stop-forward layers and the other candidates and train the model to convergence.
The training results are added to a scatterplot, naturally creating an estimate of the Pareto frontier.




'''

