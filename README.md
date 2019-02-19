# Skip-gram (Word2Vec)

The goal of this project is to implement a skip-gram model with negative-sampling from scratch. We used a log-likelihood function that is maximised using Mini-Batch Gradient Ascent.

Input : path to a corpus of words (text file)
Output: embedding of the words in the corpus

## Prerequisites

We used these libraries:
argparse
pandas as pd
numpy as np
re  # Used to split text into sentences
string  # Used to get all punctuation
pickle  # Used to save and load embeddings
logging  # Used to save steps in a text file instead of printing them
tqdm  # Used measure and estimate the time it takes to run the script
scipy.special (expit)  # Used to compute the gradient

## Preprocessing data

1. Text to sentences
Uploading data which can be in different types (series of sentences or a whole text like a book). Then splitting sentences using '.', '!', '?' or ';' or '\n' respectively. And finally splitting words using whitespaces after removing punctuation and non-alpha.
Removing also stopwords wasn't kept due to worse results

2. Rare words pruning	
Removing  words that appears less than minCount in the corpus. 

3. High frequency words removing
Remove words that occurs more than a ratio of the total number of words in the corpus
This method hasn't been used in the final model due a worse results. (the ratio shall be chosen in a more sophisticated way than only with a grid search, which can lead to overfitting the corpus)

## Skip Gram model

1. Positive pairs 
Fixing a window size winSize. Then generating positive pairs using target words and their contexts. 
Keeping track of the words by assigning an index for each unique word (use of a dictionary for words and another for contexts)

2. Negative pairs
For each positive pair, choosing a number 'negativeRate' of random words to take from the corpus to make some negative pairs with the target word.
Keeping track of these pairs using the two dictionaries made previously.

## Train the model

Two different methods were tested here. The second one gave us much better results.

### Method 1 : Creating one and only one embedding for each word
For a corpus containing 1000 unique words, we will compute a matrix of 1000 columns where each column contains the embedding of a unique word

### Method 2 : Creating an embedding for words and one for contexts
For a corpus containing 1000 unique words, we will compute a matrix of 1000 + 1000 words where for each unique word, we will compute its embedding as a word and its embedding as a context.

For both methods the following steps are the same:
1. Initializing \theta (vector of parameters to compute) randomly at the beginning (avoiding a vector of zeros). 
2. Choosing number of epochs and batch size.
3. Compute the gradient of the chosen objective function (formula given by 'Yoav Goldberg' and 'Omer Levy'[1])
4. Update embeddings after each batch (using one of the two methods)

## Running the model

Example of command line to execute : 
```
python skipGram.py --news/news.en-00001-of-00100.txt --model news
```
The command uses the news.en-00001-of-00100.txt as training set and saves the word embeddings in news file.

Example :
news.en-00001-of-00100.txt  file containing :
- 306 068 sentences (kept only 10 000)
- 219 305 words
- 1020658 positive pairs
- 5103290 negative pairs
- 4901 unique words 

Example of the out of the previous command

![Model training](training_model.PNG)





