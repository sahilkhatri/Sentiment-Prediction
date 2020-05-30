#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:02:24 2020

@author: sahil
"""

# import required libraries
from sentence_transformers import SentenceTransformer
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, LSTM, Conv1D, MaxPooling1D, GRU, GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import pickle
import time

"""
class Imdb contains all the functionality of loading the data, cleaning the data,
creating and loading embeddings, visual representation of words, splitting the data,
 model definition, training the model, and prediction.

"""
class Imdb:
    
    def __init__(self):            
        train_path = "dataset/train.csv"
        test_path = "dataset/test.csv"
        train_embeddings_path = "train embeddings/"
        test_embeddings_path = "test embeddings/"

        
        
        self.df_train = self.load_data(train_path)                                      # call to the load_data function 
        self.df_test = self.load_data(test_path)
        
        train_sentences, train_labels, stopword = self.data_cleaning(self.df_train)      # call to the data_cleaning function
        self.show_wordcloud(train_sentences, stopword)                        # call to show_wordcloud function 


#        test_sentences, test_labels, stopword = self.data_cleaning(self.df_test)
#        self.show_wordcloud(test_sentences, stopword)
        


        
        if (input("want to create embeddings? y/n")=='y'):              # ask user if want to create new embeddings
            self.create_embeddings(train_sentences, train_labels,train_embeddings_path)                                    # emddings creation is a time consuming so it is 
            self.create_embeddings(test_sentences, test_labels,test_embeddings_path)                                                            # better to store embeddings and reuse them
      
        train_embeddings, train_labels = self.load_embeddings(train_embeddings_path)                      # call to the load_embeddings function
        test_embeddings, test_labels = self.load_embeddings(test_embeddings_path)                      # call to the load_embeddings function
            
        
        train_embeddings = train_embeddings.reshape(25000,1,768)
        test_embeddings = test_embeddings.reshape(25000,1,768)
        train_labels = np.asarray(train_labels).reshape(25000,1)
        test_labels = np.asarray(test_labels).reshape(25000,1)   
        train_shape = train_embeddings.shape[1:]                                  # this input shape is used as input to the model_definition function

#        x_train, y_train, x_valid, y_valid, x_test, y_test = self.data_split_reshape(embeddings, labels)    #call data_split_reshape function
                               # this input shape is used as input to the model_definition function

        model = self.model_definition(train_shape)                       # call to the model_definition
        self.epochs = 50                                              # initialize epochs
        self.batch_size = 64                                            # initialize batch size
        history = self.train_model(model, train_embeddings, train_labels, test_embeddings, test_labels)   #call to the train_model function
        self.plot_results(history)                                       # call to plot_results function                        
        self.test_model(test_embeddings, test_labels)                                  # call to test_model function
        
        self.tweet_pred()
        
        """
           load_data function loads the csv file into pandas and returns the 
           pandas dataframe
        """
    def load_data(self,path):
        df = pd.read_csv(path)
        return df
    
        """
            data_cleaning forms the list of stopwords (sw). From the dataframe 
            stopwords are replaced with a space. Sentiments labels (positive 
            and negative) are label encoded. This function returns the cleaned
            reviews, encoded labels, and list of stopwords
        """
    def data_cleaning(self, dataset):
        specific_wc = ['br', 'movie', 'film']
        sw = list(set(stopwords.words('english')))      # list of english specific stopwords
        sw = sw + specific_wc
        print(sw[:5])
        print(len(sw))

        sentences = []
        labels = []
        for ind, row in dataset.iterrows():             # iterate through the dataframe
            labels.append(row['sentiment'])             # labels are appended in the empty list
            sentence = row['review']                    # each review is assigned to a variable
            sentence = sentence.replace('<br /><br />', "")
            sentence = sentence.replace('\\', "")
            for word in sw: # removing stop words
                token = " "+word+" "
                sentence = sentence.replace(token, " ") # replacing stop words with space
                sentence = sentence.replace("  ", " ")
                
            sentences.append(sentence)                  # each review with replaced stopword is then appended to the list sentences
                 

        return sentences, labels, sw                    # return sentences, encoded_labels, and stopwords (sw)


        
    """
        show_wordcloud function takes cleaned sentences and stopwords as input parameters.
        This function plots the wordcloud for all the words used in the dataset except
        for the stopwords.
    """
    def show_wordcloud(self, sentences, sw):            
        # word cloud on entire reviews
        wc = WordCloud(width = 600, height = 400, 
                       background_color ='white', 
                           stopwords = sw, 
                           min_font_size = 10, colormap='Paired_r').generate(' '.join(sentences[:100]))
        plt.imshow(wc)

    """
        show_pos_wordcloud function takes dataframe and stopwords as input parameters.
        This function plots the wordcloud for all the positive sentiments used in the dataset except
        for the stopwords.
    """

    def show_pos_wordcloud(self, df, sw):
        # word cloud on positve reviews
        pos_rev = ' '.join(df[df['sentiment']=='positive']['review'].to_list()[:10000])
        wc = WordCloud(width = 600, height = 400, 
                       background_color ='white', 
                       stopwords = sw, 
                       min_font_size = 10, colormap='GnBu').generate(pos_rev)
        plt.imshow(wc)

    """
        show_neg_wordcloud function takes dataframe and stopwords as input parameters.
        This function plots the wordcloud for all the negative sentiments used in the dataset except
        for the stopwords.
    """

    def show_neg_wordcloud(self, df, sw):
        neg_rev = ' '.join(df[df['sentiment']=='negative']['review'].to_list()[:10000])
        wc = WordCloud(width = 600, height = 400, 
                       background_color ='white', 
                       stopwords = sw, 
                       min_font_size = 10, colormap='RdGy').generate(neg_rev)
        plt.imshow(wc)


    """
        create_embeddings takes the cleaned sentences and encoded_labels as input parameters.
        Using SentenceTransformer (bert-base-nli-mean-tokens) the sentences are encoded.
        These library encoded the sentences using pretrained bert model. The embeddings
        generated using this pretrained model are of size 768 per input statement.
        Finally the embeddings and encoded_labels are stored in a pickle file. This 
        is a long process, so it is better to save the embeddings for reuability.
    """
    def create_embeddings(self, sentences, encoded_labels, path):
        embedder = SentenceTransformer('bert-base-nli-mean-tokens')
        review_embeddings = embedder.encode(sentences)
        pickle.dump(review_embeddings,open(path + "sentence-embeddings",'wb'))
        pickle.dump(encoded_labels, open(path + "label-embeddings",'wb'))

    """
        load_embeddings function is for loading the saved embeddings and the labels.
    """
    def load_embeddings(self, path):
        review_embeddings = pickle.load(open(path + "sentence-embeddings",'rb'))
        encoded_labels = pickle.load(open(path + "label-embeddings",'rb'))
        review_embeddings = np.asarray(review_embeddings)
        return review_embeddings, encoded_labels
    
    """
        data_split_reshape takes loaded embeddings and labels as input parameters.
        This function then splits the data into training, validation, and testing.
        There are total 50000 entries. So i have given 37500 entries for training,
        10000 for validation, and 2500 for testing.
        After spliting the data, it needs to be reshaped so that we can give it
        as input to out machine learning model. This function finally returns the 
        reshaped train, validation, and test data.
    """
    def data_split_reshape(self, embeddings, labels):
        train_sentences = embeddings[:37500]
        train_labels = labels[:37500]
        validation_sentences = embeddings[37500:47500]
        validation_labels = labels[37500:47500]
        test_sentences = embeddings[47500:]
        test_labels = labels[47500:]

        train_sentences = train_sentences.reshape(37500,1,768)
        validation_sentences = validation_sentences.reshape(10000,1,768)
        test_sentences = test_sentences.reshape(2500,1,768)
        train_labels = train_labels.reshape(37500,1)
        validation_labels = validation_labels.reshape(10000,1)
        test_labels = test_labels.reshape(2500,1)
        
        return train_sentences, train_labels, validation_sentences, validation_labels,test_sentences, test_labels
                
    
    """
        model_definition function takes the input_shape(shape of the training data) 
        as input parameter. 
        Model architecture:
            Sequential
            Lstm 64
            Lstm 32
            GlobalAveragePooling1d
            Dense 10    activation: relu
            Dense 3     activation: relu
            Dense 1     activation: sigmoid
            optimizer: adam
            loss:   binary_crossentropy
        This function returns the model defined
    """
    def model_definition(self, input_shape):

        np.random.seed(1)
        tf.random.set_seed(2)

        lstm_model = Sequential()
        lstm_model.add(LSTM(64, input_shape=(input_shape), return_sequences=True))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(32, return_sequences=True))
        lstm_model.add(GlobalAveragePooling1D())
        lstm_model.add(Dense(10, activation = 'relu'))
        lstm_model.add(Dropout(0.2))                     
        lstm_model.add(Dense(3, activation = 'relu'))
        lstm_model.add(Dense(1, activation = 'sigmoid'))
        
        adam = optimizers.Adam(learning_rate=0.0001)
        lstm_model.compile(loss='binary_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])               
               
        lstm_model.summary()
        return lstm_model


    """
        train_model function takes model, train data, and validation data as
        input. This function trains the model using training and validation data.
        LearningRateScheduler is used to dynamically change the learning rate,
        based on the epoch value. Finally the model is saved and the object of 
        model.fit is returned.
    """

    def train_model(self, model, x_train, y_train, x_valid, y_valid):

        def scheduler(epoch):
            if epoch < 5:
                print(epoch)
                return 0.01
            else:
                return 0.01 * tf.math.exp(0.01 * (10 - epoch))


        num_epochs = self.epochs
        batch = self.batch_size
        callback = LearningRateScheduler(scheduler)
        
        
        NAME = "TENSORBOARD-IMDB-{}".format(int(time.time()))
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

        cb_list = [callback, tensorboard]
        history = model.fit(x_train, y_train, 
                                 epochs=num_epochs,
                                 callbacks = cb_list,
                                 verbose=1, 
                                 batch_size=batch,
                                 validation_data=(x_valid, y_valid))

        MODEL_NAME = "GRU-BATCH-{}-EPOCHS-{}".format(batch, num_epochs)
        model.save('saved models/{}'.format(MODEL_NAME))
        return history
    
    """
        plot_results function takes the model.fit object as input parameter.
        This function plots the accuracy and loss graphs for training and validation.
        
    """

    def plot_results(self, history):
            # loss and accuracy
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    """
        test_model function takes the test data as input. It loads the saved model,
        and evaluate the model on the test data. And prints the test accuracy and
        test loss.
    """
    def test_model(self, x_test, y_test):
        MODEL_NAME = "GRU-BATCH-64-EPOCHS-50"
        model = tf.keras.models.load_model('saved models/'+MODEL_NAME)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print('Test Accuracy : {}'.format(acc))
        print('Test loss : {}'.format(loss))
        
        
    def tweet_pred(self):
        tweet = pd.read_json("tweets/corona_april.json")  # load tweets dataset
        tweet_temp = tweet[:10]
        sw = list(set(stopwords.words('english')))      # list of english specific stopwords
        print(sw[:5])
        print(len(sw))

        sentences = []
        for ind, row in tweet_temp.iterrows():             # iterate through the tweets dataframe
            sentence = row['text']                    # each tweet is assigned to a variable
            sentence = sentence.replace('<br /><br />', "")
            sentence = sentence.replace('\\', "")
            for word in sw: # removing stop words
                token = " "+word+" "
                sentence = sentence.replace(token, " ") # replacing stop words with space
                sentence = sentence.replace("  ", " ")
                
            sentences.append(sentence)                  # each tweet with replaced stopword is then appended to the list sentences
                 
        
        embedder = SentenceTransformer('bert-base-nli-mean-tokens')
        review_embeddings = embedder.encode(sentences)          # create bert embeddings for the tweets
        review_embeddings = np.asarray(review_embeddings).reshape(len(tweet_temp), 1, 768)
        
        """
            load the trained model to predict the sentiment of the tweets
        """
        MODEL_NAME = "GRU-BATCH-64-EPOCHS-50"
        model = tf.keras.models.load_model('saved models/'+MODEL_NAME)
        pred = model.predict(review_embeddings)
        print(pred)
        predictions = []
        for i in pred:
            if i<0.5:
                predictions.append(0)       # 0 for negative sentiment
            elif i>=0.5:
                predictions.append(1)       # 1 for positive sentiment
        print(predictions)


        """
            showing the predicted sentiment result corrosponding to the tweets 
            in the form of dataframe
        """
        col = ['text','sentiment']
        pred_df = pd.DataFrame(columns=col, index=range(len(tweet_temp)))
        pred_df['text']=tweet_temp['text']
        pred_df['sentiment']=predictions
        print("### TWEET RESULT ###")
        print("1 for POSITIVE, 0 for NEGATIVE")
        print(pred_df)    
            
            
Imdb()  #class instantiation







