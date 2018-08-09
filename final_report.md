# Similarity Learning using Neural Networks

## Index
1. Problem Statement
2. Similarity Learning Tasks
3. Evaluation Metrics
4. Establishing Baselines
5. About Datasets
6. The journey
7. Notes on Finetuning Models
8. Thoughts on Deep Learning Models
9. Conclusion
10. Future Work


## 1. Problem Statement
The task of Similarity Learning is a task of learning similarities or disimilarites between sentences/paragraphs. The core idea is to find some sort of vector representation for a sentence/document/paragraph. This vector representation can then be used to compare a given sentence with other sentences. For example<br/>
Let's say we want to compare the sentences: `"Hello World"`, `"Hi there friend"` and `"Bye bye"`.  
We have to develop a function, `similarity_fn` such that<br/>


	similarity_fn("Hello there", "Hi there friend") > similarity_fn("Hello there friend", "Bye Bye")
	similarity_fn("Hello there", "Hi there friend") = 0.9
	similarity_fn("Hello there friend", "Bye Bye") = 0.1


As such, Similarity Learning does not require Neural Networks (tf-idf and other such sentence representations can work as well). However, with the recent domination of Deep Learning in NLP, newer models have been performing the best on Similarity Learning tasks.

The idea of finding similarities can be used in the following ways [1]:<br/>
**1. Regresssion Similarity Learning:** given `document1` and `document2` we need a `similiarity measure` as a float value.

**2. Classification Similarity Learning:** given `document1` and `document2` we need to classify them as similar or disimilar

**3. Ranking Similarity Learning:** given `document`, `similar-document` and `disimilar-document`, we want to learn a `similarity_function` such that `similarity_function(document, similar-document)  > similarity_function(document, disimilar-document)`

**4. Locality Sensitivity Hashing:** hashes input items so that similar items map to the same "buckets" in memory with high probability.


<br/>Usually, the similarity can be measures like:
1. A vector representation is calculated for the two documents and the cosine similarity between them is their similarity score
2. The two documents are made to interact in several different ways (neural attention, matrix multiplication, LSTMs) and a softmax probablity is calculated of them being similar or dissimilar. Example: `[0.2, 0.8] -> [probablity of being dissimilar, probablity of being similar]`

## 2. Similarity Learning Tasks
**1. Question Answering/Information Retrieval**  
Given a question and a number of candidate answers, we want to rank the answers based on their likelihood of being the answer to the question.
Example:

	Q: When was Gandhi born?
	A1: Gandhi was a freedom fighter
	A2: He was born 2 October, 1989
	A3: He was a freedom fighter

The answer selected by the model is  
`argmax_on_A(simialrity_fn(Q, A1), simialrity_fn(Q, A2), simialrity_fn(Q, A3)) = A2`

Conversation/Respone Selection is similar to Question Answering, for a given Question `Q`, we want to select the most appropriate response from a pool of responses (`A1, A2, A3, ...`)
<br/> Example datasets: [WikiQA](https://www.microsoft.com/en-us/download/confirmation.aspx?id=52419)

**2. Textual Entailment**  
Given a text and a hypothesis, the model must make a judgement to classify the text
<br/>Example from [SNLI](https://nlp.stanford.edu/projects/snli/)

	- "A man selling donuts to a customer during a world exhibition event held in the city of Angeles" -- "A man selling donuts to a customer." --> Entailment
	- "Two young boys of opposing teams play football, while wearing full protection uniforms and helmets." -- "boys scoring a touchdown" --> Neutral
	- "Under a blue sky with white clouds, a child reaches up to touch the propeller of a plane standing parked on a field of grass." -- "A child is playing with a ball." --> Contradiction

**3. Duplicate Document Detection/Paraphrase Detection**  
Given 2 docs/sentences/questions we want to predict the probablity of the two questions being duplicate<br/>
Example:

	- "What is the full form of AWS" -- "What does AWS stand for?" --> Duplicate
	- "What is one plus one?" -- "Who is president?" --> Not Duplicate
Example Dataset: [Quora Duplicate Question Dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)

**4. Span Selection:**  
Given a context passage and a question, we have to find the span of words in the passage which answers the question.
<br/> Example:

	passage: "Bob went to the doctor. He wasn't feeling too good"
	question: "Who went to the doctor"
	span: "*Bob* went to the doctor. He wasn't feeling too good"

Example Dataset: [SQUAD](https://rajpurkar.github.io/SQuAD-explorer/)

## 3. Evaluation Metrics
While some similarity functions have simple metrics(Accuracy, Precision), it can get a bit complicated/different for others.

**Duplicate Question/Paraphrase Detection and Textual Entailment use Accuracy.** <br/>Since it is easy to say that an answer is either correct or incorrect. Accuracy is simply the ratio of correct answers to the total number of questions.

**Span Selection uses Exact Match and F1 score** <br/> How many times did the model predict the correct span?

(A full list of datasets and metrics can be found [here](https://github.com/aneesh-joshi/aneesh-joshi.github.io/blob/master/_posts/2018-07-18-dataset-description.markdown))

It gets a bit tricky when you come to evaluate Question Answering systems. The reason for this is: The results given by these systems is usually in an ordereing of the candidate documents.<br/>
For a query `Q1` and set of candidate answers (`very-relevant-answer`, `slightly-relevant-answer`, `irrelevant-answer`)<br/>
The evaluation metric should score the ordering of <br/>

	(1: very-relevant-answer, 2: slightly-relevant-answer, 3:irrelevant-answer) > (1: slightly-relevant-answer, 2: very-relevant-answer, 3:irrelevant-answer) > (1: irrelevant-answer, 2: slightly-relevant-answer, 3:irrelevant-answer)

Basically, the metric should give a higher score to an ordering where the more relevant documents come first and less relevant documents come later.

The metric commonly used for this are Mean Average Precision(MAP), Mean Reciprocal Rank(MRR) and Normalized Discounted Cumulative Gain(nDCG).

### Mean Average Precision (MAP)
It is a measure to be calculated over a full dataset. A dataset like WikiQA, will have a setup like:

	q_1, (d1, d2, d3)
	q_2, (d4, d5)
	q_3, (d6, d7, d8, d9, d10)
	.
	.
	.

we find the average precision for every `q_i` and then take the mean over all the `q`s

Mean (Over all the queries) Average Precision(Over all the docs for a query)

Some psuedo code to describe the idea:

	num_queries = len(queries)
	precision = 0
	for q, candidate_docs in zip(queries, docs):
	    precision += average_precision(q, candidate_docs)
	mean_average_precision = precision / num_queries

I will solve an example here:

	Model predictions: (q1-d1, 0.2), (q1-d2, 0.7) , (q1-d3, 0.3)
	Ground Truth     : (q1-d1, 1),   (q1-d2, 1) ,     (q1-d3, 0)

We will sort our predictions based on our similarity score (Since our ordering/ranking is in that order)
	
	(d2, 0.7, 1), (d3, 0.3, 0), (d1, 0.2, 1)


	Average Precision = (Sum (Precision@k * relevance@k) for k=1 to n) / Number of Relevant documents

(TODO  Insert Image Formula)

<br/>where<br/>

	n : the number of candidate docs
	relevance@k : 1 if the kth doc is relevant, 0 if not
	Precision@k : precision till the kth cut off where precision is : (number of retrieved relevant docs / total retrieved docs)

So, `Average Precision = (1/1 + 0 + 2/3)/2 = 0.83`

Here's how:

At this point, we can discard our score (since we've already used it for ordering)
	
	(d2, 1) (d3, 0) (d1, 1)


Number of Relevant Documents is `(1 + 0 + 1) = 2`

Precision at 1:
We only look at (d2, 1)

	p@1 = (number of retrieved relevant@1/total number of retrieved docs@1) * relevance@1
	    = (1 / 1) * 1
	    = 1

Precision at 2:
We only look at (d2, 1) (d3, 0)

	p@2 = (number of retrieved relevant@2/total number of retrieved docs@2) * relevance@2
	    = (1 / 2) * 0
	    = 0

Precision at 3:
We only look at (d2, 1) (d3, 0) (d1, 1)

	p@3 = (number of retrieved relevant@3/total number of retrieved docs@3) * relevance@3
	    = (2 / 3) * 1
	    = 2/3

Therefore, Average Precision for q1 

									= (Precision@1 + Precision@2 + Precision@3) / Number of Relevant Documents
                                    = (1 + 0 + 2/3) / 2
                                    = 0.83

To get Mean Average Precision, we calculate Average Precision for all `q`s in it and average it/take the mean.

### Normalized Discounted Cumulative Gain
[This](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) Wikipedia Article is a great source for understanding NDCG.
First, let's understand Discounted Cumulative Gain.
> The premise of DCG is that highly relevant documents appearing lower in a search result list should be penalized as the graded relevance value is reduced logarithmically proportional to the position of the result.

`DCG@P = Sum for i=1 to P (relevance_score(i) / log2(i + 1))`

Let's take the previous example:  
`(d2, 0.7, 1), (d3, 0.3, 0), (d1, 0.2, 1)`

`DCG@3 = 0.7/log2(1 + 1) + 0.3/log2(1 + 2) + 0.2/log2(1 + 3)`

As can be seen, if a high score is given

**For getting the Normalized Discounted:**
> Since result set may vary in size among different queries or systems, to compare performances the normalised version of DCG uses an ideal DCG. To this end, it sorts documents of a result list by relevance, producing an ideal DCG at position p ( {\displaystyle IDCG_{p}} IDCG_p), which normalizes the score:

nDCG = DCG / IDCG

### Should I use MAP or NDCG?
Use MAP when you a candidate document is either relevant or not relevant (1 or 0)
Use nDCG when you have a range of values for relevance. For example, when documents have relevance `(d1 : 1, d2: 2, d3: 3)`

For a more detailed idea, refer to [this](https://github.com/faneshion/MatchZoo/issues/109) issue on the MatchZoo repo.


As per [@bwanglzu](https://github.com/bwanglzu)
>@aneesh-joshi I'd say it can be evaluated with nDCG, but not meaningful or representative.
For binary retrieval problem (say the labels are 0 or 1 indicates relevant or not), we use P, AP, MAP etc.
For non-binary retrieval problem (say the labels are real numbers greater equal than 0 indicates the "relatedness"]), we use nDCG.
In this situation, clearly, mAP@k is a better metric than nDCG@k.
In some cases for non-binary retrieval problem, we use both nDCG & mAP such as microsoft learning to rank challenge link.


#### About implementing MAP
While the idea of the metric seems easy, there are varying implementations of it which can be found below in the section "MatchZoo Baselines". Ideally, the most trusted metric implementations should be the one provided by [TREC](https://en.wikipedia.org/wiki/Text_Retrieval_Conference) and I would imagine that's what people use when publishing results.

###### About TREC and trec_eval
The **T**ext**Re**trieval**C**onference(TREC) is an ongoing series of workshops focusing on a list of different information retrieval (IR) research areas, or tracks. They provide a standard method of evaluating QA systems called `trec_eval`. It's a C binary whose code can be found [here](https://github.com/usnistgov/trec_eval)


Please refer to [this blog](http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system) to get a complete understanding of trec

Here's the short version:  
As the above tutorial link will tell, you will have to download the TREC evaluation code from https://trec.nist.gov/trec_eval/
run `make` in the folder and it will produce the `trec_eval` binary.

The binary requires 2 inputs:
1. **qrels:** It is the query relevance file which will hold the correct answers.
2. **pred file:** It is the predicition file which will have the predictions made by your model

It can be run like:  
`$ ./trec_eval qrels pred`

The above will provide some standard metrics. You can also specify metrics on your own like:  
`$ ./trec_eval -m map -m ndcg_cut.1,3,5,10,20 qrels pred`

The above will provide MAP value and nDCG at cutoffs of 1, 3, 5, 10 and 20

The qrels format is:

    Format
    ------
    <query_id>\t<0>\t<document_id>\t<relevance>

    Note: parameter <0> is ignored by the model

    Example
    -------
    Q1  0   D1-0    0
    Q1  0   D1-1    0
    Q1  0   D1-2    0
    Q1  0   D1-3    1
    Q1  0   D1-4    0
    Q16 0   D16-0   1
    Q16 0   D16-1   0
    Q16 0   D16-2   0
    Q16 0   D16-3   0
    Q16 0   D16-4   0

The model's prediction, `pred` should be like:

    Format
    ------
    <query_id>\t<Q0>\t<document_id>\t<rank>\t<model_score>\t<STANDARD>

    Note: parameters <Q0>, <rank> and <STANDARD> are ignored by the model and can be kept as anything
    I have chose 99 as the rank. It has no meaning.

    Example
    -------
    Q1  Q0  D1-0    99  0.64426434  STANDARD
    Q1  Q0  D1-1    99  0.26972288  STANDARD
    Q1  Q0  D1-2    99  0.6259719   STANDARD
    Q1  Q0  D1-3    99  0.8891963   STANDARD
    Q1  Q0  D1-4    99  1.7347554   STANDARD
    Q16 Q0  D16-0   99  1.1078827   STANDARD
    Q16 Q0  D16-1   99  0.22940424  STANDARD
    Q16 Q0  D16-2   99  1.7198141   STANDARD
    Q16 Q0  D16-3   99  1.7576259   STANDARD
    Q16 Q0  D16-4   99  1.548423    STANDARD

## 4. Establishing Baselines
In our journey of developing a similarity learning model, we first decided to use the [WikiQA](https://www.microsoft.com/en-us/download/confirmation.aspx?id=52419) dataset since it is a well studied and common baseline used in many papers like [SeqMatchSeq](https://arxiv.org/pdf/1611.01747.pdf), [BiMPM](http://aclweb.org/anthology/P17-2081) and [QA-Transfer](https://arxiv.org/pdf/1702.03814.pdf).


### Word2Vec Baseline
To understand the level of improvement we could make, we set up a baseline using word2vec. We use the average of word vectors in a sentence to get the vector for a sentence/document.

	"Hello World" -> (vec("Hello") + vec("World"))/2

When 2 documents are to be compared for similarity/relevance, we take the Cosine Similarity between their vectors as their similarity. (300 dimensional vectors were seen to perform the best, so we chose them.)

The w2v 300 dim MAP score on the full set(100%) of WikiQA is 0.59<br/> train split(80%) of WikiQA is 0.57<br/> test split(20%) of WikiQA is 0.62<br/> dev split(10%) of WikiQA is 0.62<br/>

### MatchZoo Baselines
I had found a repo called [MatchZoo](https://github.com/faneshion/MatchZoo) which had almost 10 different Similarity Learning models developed and benchmarked. According to their benchmarks, their best model performed around 0.65 MAP on WikiQA.

I tried to reproduce their results using their scripts but found that the scores weren't keeping up with what they advertised. I raised an [issue](https://github.com/faneshion/MatchZoo/issues/103). (There was a similar issue before mine as well.) They went about fixing it. I was later able to reproduce their results (by reverting to a point earlier in the commit history) but there was now a lack of trust in their results. I sought out to write my own script in order to make my evaluation independent of theirs. The problem was that they had their own way of representing words and other such internal functions. Luckily, they were dumping their results in the TREC format. I was able to scrape that and evaluate it using [this](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/06af2545b3034fb28a125233ed7722d128855791/old_stuff/evaluate_models.py#L248) script.

I developed a table with my baselines as below:

![alt](https://github.com/aneesh-joshi/aneesh-joshi.github.io/blob/master/_posts/images/ranged%20benchmarks%20mz.PNG)

**Note:** the values are different from the established baseline because there is some discrepancy on how MAP should be calculated. I initially used my implementation of it for the table. Later, I moved to that provided by trec.
This is how I implement it:

	def mapk(Y_true, Y_pred):
	    """Calculates Mean Average Precision(MAP) for a given set of Y_true, Y_pred

	    Note: Currently doesn't support mapping at k. Couldn't use only map as it's a
	    reserved word

	    Parameters
	    ----------
	    Y_true : numpy array or list of ints either 1 or 0
	        Contains the true, ground truth values of the relevance between a query and document
	    Y_pred : numpy array or list of floats
	        Contains the predicted similarity score between a query and document

	    Examples
	    --------
	    >>> Y_true = [[0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]]
	    >>> Y_pred = [[0.1, 0.2, -0.01, 0.4], [0.12, -0.43, 0.2, 0.1, 0.99, 0.7], [0.5, 0.63, 0.92]]
	    >>> print(mapk(Y_true, Y_pred))
	    0.75
	    """

	    aps = []
	    n_skipped = 0
	    for y_true, y_pred in zip(Y_true, Y_pred):
	        # skip datapoints where there is no solution
	        if np.sum(y_true) < 1:
	            n_skipped += 1
	            continue

	        pred_sorted = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)
	        avg = 0
	        n_relevant = 0

	        for i, val in enumerate(pred_sorted):
	            if val[0] == 1:
	                avg += 1. / (i + 1.)
	                n_relevant += 1

	        if n_relevant != 0:
	            ap = avg / n_relevant
	            aps.append(ap)
	    return np.mean(np.array(aps))


This is how it's implemented in MatchZoo:

	def map(y_true, y_pred, rel_threshold=0):
	    s = 0.
	    y_true = _to_list(np.squeeze(y_true).tolist())
	    y_pred = _to_list(np.squeeze(y_pred).tolist())
	    c = list(zip(y_true, y_pred))
	    random.shuffle(c)
	    c = sorted(c, key=lambda x:x[1], reverse=True)
	    ipos = 0
	    for j, (g, p) in enumerate(c):
	        if g > rel_threshold:
	            ipos += 1.
	            s += ipos / ( j + 1.)
	    if ipos == 0:
	        s = 0.
	    else:
	        s /= ipos
	    return s

The above has a random shuffle because it prevents false numbers. Refer to [this](https://github.com/faneshion/MatchZoo/issues/232) issue.

After studying these values, we decided to implement the DRMM_TKS model since it had the best score for less tradeoffs (like time to train and memory)

**Retrospection Point:** It should've hit us at this point that the best MAP score of 0.65 as compared to the baseline of 0.58 (of word2vec) isn't so much more (only 0.07). In the later weeks it became clearer that we wanted to develop a model which could perform at least 0.75  (i.e., 0.15 more than the word2vec baseline.) On changing the way we evaluated (shift from my script to trec), the w2v baseline went upto 0.62. Which meant, we need a MAP of at least 0.78 to be practically useful. Moreover, the very [paper](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/) which introduced WikiQA claimed a MAP of 0.65. We should've realized at this point that investigating with these models was probably pointless(since the best they had so far was merely 0.65). Maybe, we hoped that finetuning these models would lead to a higher score.


### Non standard things about Similarity Learning Models
There are some things which area bit non-standard when it comes to Similarity Learning. These are the things one wouldn't know while working on standard Deep Learning things.  
The first one is evaluating with MAP, nDCG, etc which I have already explained above.
The second is the 

### loss function.
Ideally, we would like to use a loss function which would improve our metric (MAP). Unforunately, MAP isn't a differentiable function. It's something like saying, I want to increase Accuracy, let's make that the loss function. In the Similarity Learning context, we use the `Rank Hinge Loss`
This loss function is especially good for Ranking Problems. More details can be found [here](https://en.wikipedia.org/wiki/Hinge_loss)

`loss = maximum(0., margin + y_neg - y_pos)` where margin is set to 1 or 0.5

Let's look at the keras definitions to get a better idea of the intuition

	def hinge(y_true, y_pred):
	    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)

In this case, the y_true can be {1, -1}
For example, y_true = [1 , -1, 1] , y_pred = [0.6, 0.1, -0.9]

	K.mean(K.maximum(1 - [1, -1, 1] * [0.6, 0.1, -0.9], 0))
	= K.mean(K.maximum(1 - [0.6, -0.1, -0.9], 0))
	= K.mean(K.maximum([0.4, 1.1, 1.9], 0))
	= K.mean([0.4, 1.1, 1.9])

Effectively, the more wrong answers contribute more to the loss

	def categorical_hinge(y_true, y_pred):
	    pos = K.sum(y_true * y_pred, axis=-1)
	    neg = K.max((1.0 - y_true) * y_pred, axis=-1)
	    return K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)

In this case, there is a categorization, so the true values are one hots  
For example, y_true = [[1, 0, 0], [0, 0, 1]], y_pred = [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]

	pos = K.sum([[1, 0, 0], [0, 0, 1]] * [[0.9, 0.1, 0.1], [0.7, 0.2, 0.1]], axis=-1)
	    = K.sum([[0.9, 0, 0], [0, 0, 0.1]], axis=-1)  # check the predictions for the correct answer
	    = [0.9, 0.1]
	neg = K.max((1.0 - [[1, 0, 0], [0, 0, 1]]) * [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]], axis=-1)
	    = K.max([[0, 1, 1], [1, 1, 0]] * [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]], axis=-1)
	    = K.max([[0, 0.1, 0.1], [0.7, 0.2, 0]], axis=-1)
	    = [0.1, 0.7]
	loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
	     = K.mean(K.maximum(0.0, [0.1, 0.7] - [0.9, 0.1] + 1), axis=-1)
         = K.mean(K.maximum(0.0, [0.2, 1.6]), axis=-1)
         = K.mean([0.2, 1.6])

Again, effectively, the more wrong value (0.7) contributes more to the loss, whereas the correct value(0.9) reduces it greatly

	K: is the Keras backend
	effectively K=tf
	K.maximum : is elementwise maximum
	K.max gets the maximum value in a tensor


The third "non-standard" idea is that of 

### Pairwise, Listwise and Pointwise.
[Here](https://medium.com/@nikhilbd/pointwise-vs-pairwise-vs-listwise-learning-to-rank-80a8fe8fadfd), [here](https://www.quora.com/What-are-the-differences-between-pointwise-pairwise-and-listwise-approaches-to-Learning-to-Rank) and [here](https://en.wikipedia.org/wiki/Learning_to_rank) are some good resources.

Here's my take on it:  
These 3 represent different ways of training a ranking model by virtue of the loss function.

#### Pointwise
Here we take a sentence and try to give it a rank. For example, we take the sentence (`"I like chicken wings"`) and try to give it a numerical score (`0.34`)  for the question (`"What do I like to eat?"`). 

#### Pairwise
We take two docs and try to rank them as compared to each other. "x1 is more relevant than x2"
>Given a pair of documents, they try and come up with the optimal ordering for that pair and compare it to the ground truth. The goal for the ranker is to minimize the number of inversions in ranking i.e. cases where the pair of results are in the wrong order relative to the ground truth.

#### ListWise
> Listwise approaches directly look at the entire list of documents and try to come up with the optimal ordering for it


## 5. About Datasets
Here, I will give examples of different datasets.


Dataset Name | Link | Suggested Metrics | Some Papers that use the dataset | Brief Description
-- | -- | -- | -- | --
WikiQA | <ul><li>[Dataset](https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip)</li><li>[Paper](https://aclweb.org/anthology/D15-1237)</li></ul> | <ul><li>MAP</li><li>MRR</li></ul> | <ul><li>SeqMatchSeq(`MULT` MAP=0.74, MRR=0.75)</li><li>BiMPM(MAP=0.71, MRR=0.73)</li><li>QA-Transfer(`SQUAD*` MAP=0.83, MRR=84.58, P@1=75.31)</li></ul> | Question-Candidate_Answer1_to_N-Relevance1_to_N
SQUAD 2.0 | [Website](https://rajpurkar.github.io/SQuAD-explorer/) | <ul><li>Exact Match</li><li>F1</li></ul> | QA-Transfer(for pretraining) | Question-Context-Answer_Range_in_context
Quora Duplicate Question Pairs | <ul><li>gensim-data(quora-duplicate-questions)</li><li>[Quora Official](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)</li><li>[Kaggle](https://www.kaggle.com/c/quora-question-pairs)</li></ul> | Accuracy, F1 | <ul><li>BiMPM(Acc=88.17%)</li><ul> | Q1-Q2-DuplicateProbablity
Sem Eval 2016 Task 3A | genism-data(semeval-2016-2017-task3-subtaskA-unannotated) | <ul><li>MAP</li><li>AvgRecall</li><li>MRR</li><li>P</li><li>R</li><li>F1</li><li>Acc</li> | QA-Transfer(`SQUAD*` MAP=80.2, MRR=86.4, P@1=89.1) | Question-Comment-SimilarityProbablity
MovieQA | <ul><li>[Paper](http://movieqa.cs.toronto.edu/static/files/CVPR2016_MovieQA.pdf)</li><li>[Website](http://movieqa.cs.toronto.edu/home/)</li></ul> | Accuracy | SeqMatchSeq(`SUBMULT+NN` test=72.9%, dev=72.1%) | Plot-Question-Candidate_Answers
InsuranceQA | [Website](https://github.com/shuzi/insuranceQA) | Accuracy | SeqMatchSeq(`SUBMULT+NN` test1=75.6%, test2=72.3%, dev=77%) | Question-Ground_Truth_Answer-Candidate_answer
SNLI | <ul><li>[Paper](https://nlp.stanford.edu/pubs/snli_paper.pdf)</li><li>[Website](https://nlp.stanford.edu/projects/snli/)</li></ul> | Accuracy | <ul><li>QA-Transfer(for pretraining)</li><li>SeqMatchSeq(`SUBMULT+NN` train=89.4%, test=86.8%)</li><li>BiMPM()`Ensemble` Acc=88.8%)</li></ul> | Text-Hypothesis-Judgement
TRECQA | <ul><li>https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art)</li><li>https://github.com/castorini/data/tree/master/TrecQA</li><li>http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2</li></ul> | <ul><li>MAP</li><li>MRR</li></ul> | BiMPM(MAP:0.802, MRR:0.875) | Question-Candidate_Answer1_to_N-relevance1_to_N
SICK | [Website](http://clic.cimec.unitn.it/composes/sick.html) | Accuracy | QA-Transfer(Acc=88.2) | sent1-sent2-entailment_label-relatedness_score


More dataset info can be found at the [SentEval](https://github.com/facebookresearch/SentEval) repo.


Model/Paper | Link
----------- | ----
SeqMatchSeq | https://arxiv.org/pdf/1611.01747.pdf
QA-Transfer | http://aclweb.org/anthology/P17-2081
BiMPM | https://arxiv.org/pdf/1702.03814.pdf

### Some useful examples

#### SQUAD
![SQUAD](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/squad.png)

#### SQUAD, WikiQA, SemEval, SICK
![alt](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/dataset_description.png)

#### MovieQA and InsuranceQA
![alt](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/MovieQA&InsuranceQA.png)


## 6. My Journey
After going through the initial evaluation, we decided to implement the DRMM TKS model, since it gave the best score (0.65 MAP on WikiQA) with acceptable memory usage and time for training. The first model was considerable important since it set the standard for all other models to follow. I had to learn and implement a lot of Software Engineering principles like doctest, docstrings, etc. This model also helped me set up the skeleton structure needed for any Similarity Learning model. The code can be found [here](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/master/sl_eval/models/drmm_tks.py)

### This is generally what needs to be done:

1. Iterate through the train set's sentences and develop your model's word2index dictionary(one index for each word). This is useful, especially when it comes to predicting on new data and we need to translate new sentences.
2. Translate all the words into indexes using the above dict.
3. Load a pretrained Embedding Matrix (like Glove) and reorder it so that its indices match with your word2index dictionary.
4. There will be some words in the embedding matrix which aren't there in the train set. Although these extra embeddings aren't useful when it comes to training, they will be useful while testing and validating. So, append the word2index dict with these new words and reorder the embedding matrix accordingly.
5. Make 2 additional entries in the embedding matrix for the pad word and the unkown word.
	- Since all sentences won't be of the same shape, we will inevitably have to pad the sentences to the same length. You should decide at this point whether the pad vector is a zero vector or a random vector. 
	- While testing and validating, you will inevitably encounter an unkown word (a word not in your word2index dictionary). All these unkown words should be set to the same vector. Again, you can choose between a zero vector and a random vector.
6. Once all the data is in the right format, we need to create a generator which will stream out batches of training examples (with padding added ofcourse). In the Question Answering context, there is a special consideration here. Usually, a training set will be of the form `[(q1, (d1, d2, d3), (q2, (d4, d5)))]` with the releavnce labels like `[(1, 0, 0), (0, 1)]`. While setting up training examples, it's a good idea to make training pairs like [(q, d_rel), (q, d_not_relevant), (q, d_relevant), ...]. This "should" help the model learn the correct classification by contrasting relevant and non relevant answers.
7. Now that you batch generator is spitting out batches to train on, create a model, fix the loss, optimizer and other such parameters and train.
8. The metrics usually reported by a library like Keras won't be very useful to evaluate your model as it trains. So, at the end of each epoch, you should set up your own callback as I have done [here](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/master/sl_eval/models/utils/custom_callbacks.py)
9. Once the model training is done, evaluate it on the test set. You can obviously use your own implementation, but as a standard, it would be better to write your results into the TREC format as I have done [here](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/06af2545b3034fb28a125233ed7722d128855791/evaluation_scripts/WikiQA/eval_wikiqa.py#L224) and then use the `trec_eval` binary described before for evaluating. This standardizes evaluation and makes it easier for others to trust the result.
10. Saving models can get a bit tricky, especially if you're using keras. Keras prefers it's own way of saving against the satandard gensim `util.SaveLoad` You will have to save the keras model separate from the non-keras model and combine them while loading. Moreover, if you have any Lambda Layer, you will have to replace them with custom layers like I did [here](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/master/sl_eval/models/utils/custom_layers.py) for the TopK Layer.

Most of the above has been done in [my script](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/master/sl_eval/models/drmm_tks.py) and you can mostly just use that skeleton while changing just the `_get_keras_model` function.

Once my DRMM_TKS model was set up, it was performing very poorly (0.54 on MAP). After trying every idea I could imagine for fine tuning it, I ended up with a model which performed 0.63 on MAP. Beyond this point, however, I didn't feel like any changes I made would make it perform any better. You are welcome to try.  
Seeing that DRMM_TKS wasn't doing so well, I moved my attention to the next best performing model in my list of model, [MatchPyramid](https://arxiv.org/abs/1606.04648). This model, after some tuning, managed to score 0.64-0.65 MAP.

It was around this point that we came to the realization that the score wasn't much more than the baseline and a score of `unsupervised baseline plus 0.15` would be needed to make it a worthwhile model to collect supervised data for.

##### String of events leading to a change in models
I started looking for other datasets and happened to stumble upon the [BiMPM](https://arxiv.org/abs/1702.03814) paper which claimed a score of 0.71 on WikiQA. In it's score comparison table

![alt](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/pic1.png)


it had cited another paper [SeqMatchSeq](https://arxiv.org/pdf/1611.01747.pdf) which claimed a further score of 0.74.
I went about implementing BiMPM [here](https://github.com/aneesh-joshi/gensim/blob/my_full_tune_branch/gensim/models/experimental/eval_bimpm.py) and [here](https://github.com/aneesh-joshi/gensim/blob/my_full_tune_branch/gensim/models/experimental/bimpm.py) which didn't perform as expected. I then ran the author's [original code](https://github.com/shuohangwang/SeqMatchSeq), where it unfortunately turned out that the author had reported the score on the dev set, not the test set (as per my run at least (with the author's random seeds)). (On asking the author in [this issue](https://github.com/shuohangwang/SeqMatchSeq/issues/11), the author claims the split was test.) The actual test score was 0.69, not much more than the baseline for such a complicated model. I also found a pytorch implementation of SeqMatchSeq [here](https://github.com/pcgreat/SeqMatchSeq/) in which there was an [issue](https://github.com/pcgreat/SeqMatchSeq/issues/1) about MAP coming to 0.62 instead of 0.72 on dev set. The author commented:

>I am afraid you are right. I used to reach `~` 72% via the given random seed on an old version of pytorch, but now with the new version of pytorch, I wasn't able to reproduce the result.
My personal opinion is that the model is neither deep or sophisticated, and usually for such kind of model, tuning hyper parameters will change the results a lot (although I don't think it's worthy to invest time tweaking an unstable model structure).
If you want guaranteed decent accuracy on answer selection task, I suggest you take a look at those transfer learning methods from reading comprehension. One of them is here https://github.com/pcgreat/qa-transfer



### About QA-Transfer and BiDAF
And thus, my hunt has lead me to the paper : [Question Answering through Transfer Learning from Large Fine-grained Supervision Data](http://aclweb.org/anthology/P17-2081) which makes a crazier claim on MAP : **0.83  (On Ensemble of 12)**

The paper's author provides the implementation [here](https://github.com/shmsw25/qa-transfer) in tensorflow.

The author makes some notable claims in it's abstract:

>We show that the task of question answering (QA) can significantly benefit from the transfer learning of models trained on a different large, fine-grained QA dataset. We achieve the state of the art in two well-studied QA datasets, WikiQA and SemEval-2016 (Task 3A), through a basic transfer learning technique from SQuAD. 
**For WikiQA, our model outperforms the previous best model by more than 8%.** We demonstrate that finer supervision provides better guidance for learning lexical and syntactic information than coarser supervision, through quantitative results and visual analysis. We also show that a similar transfer learning procedure achieves the state of the art on an entailment task.


So, how this model works is
It takes an existing model for QA called [BiDirectional Attention Flow (BiDAF)](https://theneuralperspective.com/2017/01/08/bidirectional-attention-flow-for-machine-comprehension/), which would take in a query and a context. It would then predict the range/span of words in the context which is relevant to the query. It was adapted to the [SQUAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

Example of SQUAD:

![SQUAD](https://raw.githubusercontent.com/aneesh-joshi/aneesh-joshi.github.io/master/_posts/images/squad.png)

The QA-Transfer takes the BiDAF net and chops off the last layer to make it more QA like, ie., something more like WikiQA:

	q1 - d1 - 0
	q1 - d2 - 0
	q1 - d3 - 1
	q1 - d4 - 0

They call this modified network BiDAF-T

Then, they take the SQUAD dataset and break the context into sentences and labels each sentence as relevant or irrelevant. This new dataset is called SQUAD-T

**How it works:**
SQUAD dataset is for span level QA

    passage: "Bob went to the doctor. He wasn't feeling too good"
    question: "Who went to the doctor"
    span: "*Bob* went to the doctor. He wasn't feeling too good"

This can be remodelled such that:

	question : "Who went to the doctor"
	doc1 : "Bob went to the doctor."
	relevance : 1 (True)

	question : "Who went to the doctor"
	doc2 : "He wasn't feeling too good"
	relevance : 0 (False)

Effectively, we get a really big good dataset in the QA domain. The converted file is almost 110 MB.

[TODO add table here]

The new model, BiDAF-T is then trained on SQUAD-T. It gets **MAP : 0.75** on WikiQA test set.  
When BiDAF-T is finetuned on WikiQA, it gets **MAP : 0.76**  
When BiDAF is trained on SQUAD, then the weights are transferred to SQUAD-T and it is further finetuned on WikiQA, it gets the **MAP : 0.79**
When it's done is an ensemble, it gets **MAP : 0.83**  

They call it Transfer Learning.

So, as such, it's not exactly a new model. It's just  an old model(BiDAF), trained on a modified dataset and then used on WikiQA and SentEval. However, the author claims that the model does suprisingly well on both of them. The author has provided their own repo.

Since, 0.79 feels like a good score, significantly (0.17) over the w2v baseline, I went about implementing it. You can find my implementation [here](TODO)
Unfortunately, the scores don't perform as well.

**Steps invovled:**  
I had to first convert the SQUAD dataset, a span level dataset, into a QA dataset. I have uploaded the new dataset and the script for conversion on my [repo](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts). I then trained the BiDAF-T model on the SQUAD-T dataset to check if it did infact reach the elusive score of 
I got 0.64 on average (getting a 0.67 at one point) but this isn't more than MatchPyramid and it has several drawbacks:

**Things to consider regarding QA Transfer/BiDAF:**  
- The original BiDAF code on SQUAD takes 20 hours to converge [[reference]](https://github.com/allenai/bi-att-flow)
- It is a very heavy model
- Since there is a QA Transfer from SQUAD, it won't work on non-english words. In that case, it's better to try an older method like BiMPM or SeqMatchSeq

**List of existing implementations**
- BiDAF code:
	- [Original Tensorflow 0.11 Implementation](https://github.com/allenai/bi-att-flow)
	- [AllenAI-DeepQA Keras Implementation (Now deprecated)](https://github.com/allenai/deep_qa)
	- [AllenAI-AllenNLP PyTorch Implementation](https://github.com/allenai/allennlp)
	- [Independent PyTorch Implementation by @jojonki](https://github.com/jojonki/BiDAF) (unable to reproduce BiDAF)
	- [Original Implementation Tensorflow 1.8 fork by @Vimos](https://github.com/Vimos/bi-att-flow/tree/tf1.8)
- BiDAF-T code:
	- [QA Transfer Original Repo (forked from BiDAF tensorflow)](https://github.com/shmsw25/qa-transfer)
	- [QA Transfer Slightly modified fork (forked from BiDAF tensorflow)](https://github.com/pcgreat/qa-transfer)
	- [My Implmentation](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/master/evaluation_scripts/WikiQA/eval_wikiqa.py) (unable to reproduce BiDAF-T)


## Benchmarked Data

### WikiQA


WikiQA test set	w2v 50 dim	w2v 100 dim	w2v 200 dim	w2v 300 dim	FT 300 dim	MatchPyramid	DRMM_TKS	MatchPyramid Random Baseline	DRMM_TKS Random Baseline	BiDAF only pretrain	BiDAF pretrain + finetune	BiDAF Random Baseline
map	0.6016	0.6148	0.6285	0.6277	0.6199	0.6463	0.6354	0.5107	0.5394	0.601	0.5948	0.3291
gm_map	0.4642	0.4816	0.4972	0.4968	0.4763	0.5071	0.4989	0.3753	0.4111	0.4538	0.4595	0.2455
Rprec	0.4318	0.4551	0.4709	0.4667	0.4715	0.5007	0.4801	0.3471	0.3512	0.4582	0.4355	0.1156
bpref	0.4251	0.4457	0.4613	0.456	0.4642	0.4977	0.4795	0.3344	0.3469	0.4502	0.4264	0.1101
recip_rank	0.6147	0.628	0.6419	0.6373	0.6336	0.6546	0.6437	0.519	0.5473	0.6169	0.6093	0.3312
iprec_at_recall_0.00	0.6194	0.6322	0.6469	0.6437	0.6375	0.6602	0.648	0.5242	0.5534	0.6209	0.6134	0.3396
iprec_at_recall_0.10	0.6194	0.6322	0.6469	0.6437	0.6375	0.6602	0.648	0.5242	0.5534	0.6209	0.6134	0.3396
iprec_at_recall_0.20	0.6194	0.6322	0.6469	0.6437	0.6375	0.6602	0.648	0.5242	0.5534	0.6209	0.6134	0.3396
iprec_at_recall_0.30	0.6146	0.6269	0.6431	0.6401	0.6314	0.6572	0.648	0.5213	0.5515	0.618	0.6082	0.3382
iprec_at_recall_0.40	0.6125	0.6269	0.6404	0.6394	0.6293	0.6537	0.6458	0.5189	0.5488	0.6141	0.6069	0.3382
iprec_at_recall_0.50	0.6125	0.6269	0.6404	0.6394	0.6293	0.6537	0.6458	0.5189	0.5488	0.6141	0.6069	0.3382
iprec_at_recall_0.60	0.5937	0.6068	0.6196	0.6219	0.6115	0.6425	0.6296	0.5073	0.5348	0.5891	0.5841	0.3289
iprec_at_recall_0.70	0.5937	0.6068	0.6196	0.6219	0.6115	0.6425	0.6296	0.5073	0.5348	0.5891	0.5841	0.3289
iprec_at_recall_0.80	0.5914	0.6039	0.6175	0.619	0.6094	0.6401	0.627	0.5049	0.5333	0.5868	0.5838	0.3263
iprec_at_recall_0.90	0.5914	0.6039	0.6175	0.619	0.6094	0.6401	0.627	0.5049	0.5333	0.5868	0.5838	0.3263
iprec_at_recall_1.00	0.5914	0.6039	0.6175	0.619	0.6094	0.6401	0.627	0.5049	0.5333	0.5868	0.5838	0.3263
P_5	0.1893	0.1951	0.1967	0.1975	0.1926	0.1967	0.1934	0.1704	0.1835	0.1852	0.1909	0.1473
P_10	0.1107	0.1111	0.1119	0.114	0.1119	0.1156	0.1152	0.1095	0.1144	0.1115	0.1128	0.1033
P_15	0.0774	0.0776	0.0787	0.0787	0.0774	0.079	0.0785	0.0787	0.0787	0.079	0.079	0.0749
P_20	0.0595	0.0597	0.0597	0.0597	0.0591	0.0599	0.0591	0.0603	0.0601	0.0601	0.0601	0.0591
ndcg_cut_1	0.4403	0.4486	0.4691	0.4587	0.4774	0.0402	0.0402	0.0402	0.0402	0.0402	0.0402	0.0402
ndcg_cut_3	0.5867	0.6077	0.6213	0.626	0.6033	0.0121	0.0121	0.0121	0.0121	0.0121	0.0121	0.0121
ndcg_cut_5	0.6417	0.6598	0.6722	0.6743	0.6593	0.006	0.006	0.006	0.006	0.006	0.006	0.006
ndcg_cut_10	0.6825	0.693	0.7055	0.7102	0.6982	0.0024	0.0024	0.0024	0.0024	0.0024	0.0024	0.0024
ndcg_cut_20	0.6993	0.7101	0.7208	0.7211	0.7115	0.0012	0.0012	0.0012	0.0012	0.0012	0.0012	0.0012


### Quora Duplicate Questions


  | Accuracy
-- | --
MatchPyramid | 69.20%
DRMM TKS | 67.05
Word2Vec Baseline | 37.02%



### SICK

  | Accuracy
-- | --
MatchPyramid | 57.67
DTKS | 56.81


### SNLI

  | Accuracy
-- | --
MatchPyramid | 53.49
DRMM_TKS | 51.2 (Recheck)

### InsuranceQA

#### MatchPyramid


MP | Test1 | Test2 | Word2Vec Baseline | Test1 Random Baseline | Test2 Random Baseline
-- | -- | -- | -- | -- | --
map | 0.8103 | 0.8003 | 0.6975 | 0.4074 | 0.41
gm_map | 0.727 | 0.7104 | 0.5793 | 0.3118 | 0.313
Rprec | 0.7124 | 0.7005 | 0.5677 | 0.2274 | 0.2294
bpref | 0.7177 | 0.7065 | 0.569 | 0.2175 | 0.2195
recip_rank | 0.8348 | 0.8269 | 0.7272 | 0.4295 | 0.4374
iprec_at_recall_0.00 | 0.8413 | 0.8316 | 0.7329 | 0.444 | 0.4501
iprec_at_recall_0.10 | 0.8413 | 0.8316 | 0.7329 | 0.444 | 0.4501
iprec_at_recall_0.20 | 0.8412 | 0.8314 | 0.7329 | 0.4437 | 0.4499
iprec_at_recall_0.30 | 0.8398 | 0.8294 | 0.7316 | 0.4411 | 0.4471
iprec_at_recall_0.40 | 0.834 | 0.8225 | 0.7241 | 0.4305 | 0.437
iprec_at_recall_0.50 | 0.8336 | 0.8224 | 0.7238 | 0.4303 | 0.4368
iprec_at_recall_0.60 | 0.7967 | 0.7853 | 0.6813 | 0.3963 | 0.3943
iprec_at_recall_0.70 | 0.7964 | 0.7851 | 0.6812 | 0.3961 | 0.3939
iprec_at_recall_0.80 | 0.7847 | 0.7741 | 0.6662 | 0.3874 | 0.3842
iprec_at_recall_0.90 | 0.7841 | 0.7731 | 0.6652 | 0.3868 | 0.3835
iprec_at_recall_1.00 | 0.7841 | 0.7731 | 0.6652 | 0.3868 | 0.3835
P_5 | 0.2642 | 0.2596 | 0.243 | 0.174 | 0.1739
P_10 | 0.1453 | 0.1441 | 0.1453 | 0.1453 | 0.1441
P_15 | 0.0969 | 0.096 | 0.0969 | 0.0969 | 0.096
P_20 | 0.0727 | 0.072 | 0.0727 | 0.0727 | 0.072
P_30 | 0.0484 | 0.048 | 0.0484 | 0.0484 | 0.048
P_100 | 0.0145 | 0.0144 | 0.0145 | 0.0145 | 0.0144
P_200 | 0.0073 | 0.0072 | 0.0073 | 0.0073 | 0.0072
P_500 | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0029
P_1000 | 0.0015 | 0.0014 | 0.0015 | 0.0015 | 0.0014



#### DRMM_TKS


DTKS | Test1 | Test2 | Word2Vec Baseline | Test1 Random Baseline | Test2 Random Baseline
-- | -- | -- | -- | -- | --
map | 0.617 | 0.5998 | 0.7055 | 0.24 | 0.2414
gm_map | 0.5024 | 0.4835 | 0.589 | 0.1879 | 0.1899
Rprec | 0.4516 | 0.4372 | 0.5773 | 0.0806 | 0.0829
bpref | 0.4532 | 0.4377 | 0.58 | 0.0728 | 0.0743
recip_rank | 0.6496 | 0.6301 | 0.7362 | 0.2476 | 0.2488
iprec_at_recall_0.00 | 0.6575 | 0.6396 | 0.7413 | 0.2682 | 0.2698
iprec_at_recall_0.10 | 0.6575 | 0.6396 | 0.7413 | 0.2682 | 0.2698
iprec_at_recall_0.20 | 0.6573 | 0.6394 | 0.7413 | 0.2676 | 0.2697
iprec_at_recall_0.30 | 0.6547 | 0.6379 | 0.7402 | 0.2659 | 0.268
iprec_at_recall_0.40 | 0.6475 | 0.6293 | 0.7319 | 0.2589 | 0.2611
iprec_at_recall_0.50 | 0.6473 | 0.6289 | 0.7317 | 0.2589 | 0.2611
iprec_at_recall_0.60 | 0.6 | 0.5825 | 0.6869 | 0.2383 | 0.2391
iprec_at_recall_0.70 | 0.5995 | 0.5822 | 0.6866 | 0.2381 | 0.2391
iprec_at_recall_0.80 | 0.5833 | 0.5694 | 0.6734 | 0.2362 | 0.2364
iprec_at_recall_0.90 | 0.5819 | 0.5688 | 0.6731 | 0.2357 | 0.2363
iprec_at_recall_1.00 | 0.5819 | 0.5688 | 0.6731 | 0.2357 | 0.2363
P_5 | 0.2352 | 0.2292 | 0.2426 | 0.0804 | 0.0856
P_10 | 0.1453 | 0.1441 | 0.1441 | 0.1453 | 0.1441
P_15 | 0.0969 | 0.096 | 0.096 | 0.0969 | 0.096
P_20 | 0.0727 | 0.072 | 0.072 | 0.0727 | 0.072
P_30 | 0.0484 | 0.048 | 0.048 | 0.0484 | 0.048
P_100 | 0.0145 | 0.0144 | 0.0144 | 0.0145 | 0.0144
P_200 | 0.0073 | 0.0072 | 0.0072 | 0.0073 | 0.0072
P_500 | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0029
P_1000 | 0.0015 | 0.0014 | 0.0014 | 0.0015 | 0.0014




## 7. Notes on Finetuning Models
Fine tuning deep learning models can be tough, considering the model runtimes and the number of parameters.  
Here, I will list some parameters to consider while tuning:
- The word vector dimensionality(50, 100, 200, 300) and source(Glove, GoogleNews)
- Dropout on different layers
- Making the Fully Connected/Dense layer deeper or wider
- Running lesser vs more epochs
- Bigger or Smaller Batch Sizes

## 8. Thoughts on Deep Learning Models
While the models seem to be doing the best currently, they are a bit difficult to reproduce. There are several reasons I can imagine for that:
- Difference in Deep Learning Libraries and their versions used in implementations may vary or be outdated. Different libraries can have varying differences in small implementation details(like random initializations) which can lead to huge differences later on and are difficult to track.
- The large number of parameters coupled with it's stochastic nature means that the model's performance can vary greatly. Include the time it takes to get one result makes tuning them very very difficult to tune.
- "When a model doesn't perform well, is it because the model is wrong or is my implementation wrong?" This is a difficult question to answer.

## 9. Conclusion
My time spent with understanding, developing and testing Neural Networks for Similarity Learning has brought out the conclusions that:
- the current methods are not significantly better than a simple unsupervised word2vec baseline.
- the current methods are unreproducible/difficult to reproduce.
This work and effort has provided:
- a method to evaluate models
- implementations of some of the current SOTA models
- a list of datasets to evaluate models on and their readers

## 10. Future Work
If anybody was to carry on with this work, I would recommend looking at the QA-Transfer model, which claims the current "State Of The Art". Although my implementation couldn't reproduce the result, I feel like there is some merit in the model. The reason for this belief is not based in QA-Transfer, but the underlying BiDAF model, which provides [two](http://demo.allennlp.org/machine-comprehension) [demos](http://allgood.cs.washington.edu:1995/) which does really well. Since there is so much discrepancies in different implementations, it would be better to use the [ported tf1.8 BiDAF](https://github.com/Vimos/bi-att-flow/tree/tf1.8)  
If you are looking to do "newer research", I would recommend you do what the authors of QA-Transfer did. They looked at the best performing model on the SQUAD 1.1 dataset, pretrained on it, removed the last layer and evaluated. Things have changed since: There is a newer better SQUAD (2.0) and that leader board has got a new "best" model. In fact, BiDAF model is no longer the best on SQUAD 1.1 or 2.0. Just take the current best model and do the pretraining!
If there's anything to take away from QA-Transfer, it's that span supervision and pretraining can be good ideas. This, especially, seems to be a new trend regarding [Language Models](http://ruder.io/nlp-imagenet/)  

## References
[1] [Wikipedia article on Similarity Learning](https://en.wikipedia.org/wiki/Similarity_learning)