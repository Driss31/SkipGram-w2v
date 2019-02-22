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

### Method 2 : Creating an embedding for words and one for contexts. Each word will have an embedding as a target word and another embedding as a context.
For a corpus containing 1000 unique words, we will compute a matrix of 1000 + 1000 words where for each unique word, we will compute its embedding as a word and its embedding as a context.
(The method is detailed in the report)

For both methods the following steps are the same:
1. Initializing \theta (vector of parameters to compute) randomly at the beginning (avoiding a vector of zeros). 
2. Choosing number of epochs and batch size.
3. Compute the gradient of the chosen objective function (formula given by 'Yoav Goldberg' and 'Omer Levy'[1])
4. Update embeddings after each batch (using one of the two methods)

## Running the model

Example of command line to execute : 
```
python skipGram.py --text train/news.en-00001-of-00100.txt --model train
```
The command uses the news.en-00001-of-00100.txt as training set and saves the word embeddings in news file.

Example :
news.en-00001-of-00100.txt  file containing :
- 50 000 sentences
- 1 094 214 words
- 5 576 574 positive pairs
- 27 882 870 negative pairs
- 14 033 unique words 

Example of the out of the previous command

```
TRAINING: #epochs: 5, learning_rate: 0.01, batch size: 500, negativeRate: 5, winSize: 7 (+- 3)
Epoch 1/5
100%|████████████████████████████████████| 11153/11153 [39:43<00:00,  4.65it/s]
Epoch 2/5
100%|████████████████████████████████████| 11153/11153 [39:49<00:00,  4.90it/s]
Epoch 3/5
100%|████████████████████████████████████| 11153/11153 [39:59<00:00,  4.50it/s]
Epoch 4/5
100%|████████████████████████████████████| 11153/11153 [39:51<00:00,  4.68it/s]
Epoch 5/5
100%|████████████████████████████████████| 11153/11153 [40:44<00:00,  4.66it/s]
```

If the script crashes, we need to decrease the value of the learning rate.

## Testing the model

To compute the similarity between two words, we used the cosine distance. The command to use is:

```
python skipGram.py --text train/news.en-00001-of-00100.txt --model train --test
```

Some examples obtained:
"woman", "girl"    : 0.9140292408030202
"woman", "man"     : 0.9003719041046656
"woman", "fish"    : 0.6480981482215702
"woman", "bicycle" : 0.5525813805008297
"woman", "green"   : 0.41096331164361916
"woman", "red"     : 0.4022941792833647
"woman", "grizzly" : (Out of Vocabulary -> Grizzly) 0.2520814392860999

We computed the similarity between the word "president" and all the words of the corpus, and printed its 10 most similar words :
```
Similar words for president :
- prime : 0.8675725528021508
- minister : 0.8577274635745533
- barack : 0.8476139485196841
- vice : 0.8338346066461706
- senator : 0.8303469441871172
- presidentelect : 0.8234193469697527
- bush : 0.8217018577730855
- governor : 0.8114519753665658
- john : 0.8096632753090216
- hillary : 0.7903915606521152
```

10 most similar words to "financial":
```
Similar words for financial :
- economic : 0.8605254987941997
- banking : 0.8543459085572801
- housing : 0.8391836329291854
- global : 0.8367757803069268
- institutions : 0.8249920995141922
- development : 0.7974591339002757
- crisis : 0.7927191095780013
- energy : 0.7923018373003
- markets : 0.7898427369024803
- european : 0.7891532793944722
```

10 most similar words to "city":
```
Similar words for city :
- london : 0.8362202448038254
- virginia : 0.8227038534431752
- west : 0.8210187852324354
- town : 0.8084971923032622
- california : 0.8023458654379504
- iowa : 0.8014968294330261
- angeles : 0.7993281566786211
- park : 0.7968483879597993
- lake : 0.7931283454758526
- southwest : 0.7924832761941711
```


10 most similar words to "barack":

```
Similar words for barack :
- clinton : 0.8967130891694537
- obama : 0.873172647247945
- hillary : 0.8581969823452125
- president : 0.8476139485196841
- mccain : 0.8474739064219333
- presidentelect : 0.8384489353566097
- bush : 0.8324913896640769
- nominee : 0.8144799174125398
- candidate : 0.8100265607338628
- senator : 0.8096392472156546
```

## References
[1] Yoav Goldberg and Omer Levy _word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method_. 2014