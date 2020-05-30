# Sentiment-Prediction
Sentiment prediction using the bert embeddings


In this project, I trained my model on the IMDB movie review dataset. Dataset can be downloaded
from https://ai.stanford.edu/~amaas/data/sentiment/ . dataset_generate.py file creates the shuffled 
csv file for train and test data. Data of this csv file is then used in main.py file for creating
embeddings.

For creating the embeddings, I used the "bert-base-nli-mean-tokens" provided by "sentence_transformers" 
library. This creates the embeddings of size 768 for each sentence. These embeddings are then stored for 
reusability. You can play around and try various architectures in the model_definition(). 

Once the model is trained on the IMDB movie dataset, I predicted the sentiment for the tweets.
For scraping this tweets I used the "twitterscraper" library. It provides many filters to gather tweets as per our requirement. It gives the json file as output. Details for this library can be found at https://github.com/taspinar/twitterscraper.

(view this is raw format)

Recommended directory hierarchy:
	- dataset
		-dataset_generate.py
		-train.csv
		-test.csv
	- logs (tensorboard related files for visualization of model's performance)
	- saved models (model is saved here)
	- train embeddings (bert embeddings for the train sentences and its corresponding labels are stored here)
	- test embeddings (bert embeddings for the test sentences and its correspondings labels are stored here)
	- tweets
		-tweets.json
	- main.py
