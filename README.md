# Project 3 - Elliot Richardson

*Please see presentation slides in the repository.*

## Table of Contents:

I. [Problem Statement](#I-Problem-Statement)

II. [Data Dictionary](#II-Data-Used)

III. [Methodology](#III-Methodology)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   a. [Acquisition](#a-Data-acquisition) 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   b. [Processing](#b-Processing) 
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   c. [Exploratory analysis](#c-Exploratory-analysis)
   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   d. [Modeling](#d-Modeling)
   
IV. [Conclusion](#IV-Conclusion)


## I. Problem Statement

For this project, I'm an analyst at a political targeting firm. In light of the pandemic, a larger share of direct voter contact is over the internet, rather than in person. However, it isn't easy to connect social media users to their voter file records and therefore their ideological support and voter turnout scores as determined by other models. So, in order to identify persuadable targets and activatable supporters, my firm wants to create a model that can differentiate between language used by people on varying ends of the political spectrum. Eventually, they will use this model to identify target accounts on platforms like Twitter to whom campaigns will conduct outreach. 

**Question I'm seeking to answer:** Do Reddit users on varying ends of the economic political spectrum use vocabularies distinct enough for a model to differentiate between them?


## II. Data Used

Titles and content of posts requested from [r/Socialism](https://www.reddit.com/r/DemocraticSocialism/) and [r/Capitalism](https://www.reddit.com/r/Capitalism/) using the [Pushshift API](https://pushshift.io/).

### Data Dictionary:

|Feature|Type|Description|
|---|---|---|
|subreddit_s|int64|Binary indicator of the subreddit from which a post was pulled (1 = r/Socialism, 0 = r/Capitalism)|
|sentences|int64|Number of sentences in post as determined by nltk's PunktSentenceTokenizer|
|avg_sent_len|float64|Average number of words in each sentences of the post|
|words|int64|Total number of words in the post|
|avg_word_len|float64|Average number of characters in each word of the post|
|ADJ_prop|float64|Proportion of the content made up of adjectives|
|ADP_prop|float64|Proportion of the content made up of adpositions|
|ADV_prop|float64|Proportion of the content made up of adverbs|
|AUX_prop|float64|Proportion of the content made up of auxiliary words|
|CCONJ_prop|float64|Proportion of the content made up of coordinating conjunctions|
|DET_prop|float64|Proportion of the content made up of determiners|
|INTJ_prop|float64|Proportion of the content made up of interjections|
|NOUN_prop|float64|Proportion of the content made up of nouns|
|NUM_prop|float64|Proportion of the content made up of numbers|
|PART_prop|float64|Proportion of the content made up of particles|
|PRON_prop|float64|Proportion of the content made up of pronouns|
|PROPN_prop|float64|Proportion of the content made up of proper nouns|
|PUNCT_prop|float64|Proportion of the content made up of punctuation|
|SCONJ_prop|float64|Proportion of the content made up of subordinating conjunction|
|SPACE_prop|float64|Proportion of the content made up of spaces|
|SYM_prop|float64|Proportion of the content made up of symbols|
|VERB_prop|float64|Proportion of the content made up of verbs|
|X_prop|float64|Proportion of the content made up of uncategorizable words|
|vader_neg|float64|Average negativity score for the words in the post as determined by vaderSentiment's SentimentIntensityAnalyzer|
|vader_pos|float64|Average positivity score for the words in the post as determined by vaderSentiment's SentimentIntensityAnalyzer|
|vader_neu|float64|Average neutrality score for the words in the post as determined by vaderSentiment's SentimentIntensityAnalyzer|
|vader_compound|float64|Average compound score for the words in the post as determined by vaderSentiment's SentimentIntensityAnalyzer|

Check the documentation for [spaCy](https://spacy.io/api/annotation) and [vaderSentiment](https://github.com/cjhutto/vaderSentiment#resources-and-dataset-descriptions) for more information.

## III. Methodology

### a. Data acquisition

In order to acquire content for this project, I used the [Pushshift API](https://pushshift.io/) to request posts from two subreddits, r/Socialism and r/Capitalism. Because the Pushshit API allows a maximum of 100 posts for each request, I created a function that would take in a subreddit, a number, and a list to extend if one already existed, and extract 100 posts that number of times into that list. In order to be respectful of Reddit's servers, I added a time cushion of 5 seconds in between each request so I wouldn't be inundating the servers with consecutive requests. I utilized a while loop to request ~2800 posts from each subreddit that seemed to have substantive content (i.e. the body didn't read "deleted" or ""). I then combined these into one dataframe by narrowing the columns to only the ones they had in common and adding the `subreddit_s` column to differentiate between the two sets. Then I was ready to start processing!

### b. Processing

Before I altered the content in any way, I wanted to tokenize each post into sentences and words to get the `sentences`, `avg_sent_len`, `words`, and `avg_word_len` columns. After that, I converted all the text into lowercase and removed punctuation to make the content a bit more uniform across the datset. I used spaCy to parse out posts by parts of speech and create the `_prop` columns to be used in the models later. Then I used vaderSentiment's SentimentIntensityAnalyzer to create sentiment scores 
for each post. Then I moved on to feature selection in terms of which words I would include in my model. 

To start, I used sklearn's CountVectorizer to get a dataframe of unique words and the frequency of their use. I couldn't use all 26,518 unique words in these posts (not including [stop words](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)) as features in my models without waiting hours for them to run and probably overfitting to my data. So in order to narrow it down, I used nltk's PorterStemmer to group similar words together and replaced the words in the content with their stemmed versions. Then I concatenated all posts into two very long strings corresponding to each subreddit and used sklearn's TfidfVectorizer to glean the importance of each of these words to their respective "documents" (the super long strings!). I narrowed my list of stems to the top 1,000 most important words from each subreddit, which only came out to 1197 words total because there was so much overlap. Then I used CountCectorizer again to get a new DataFrame with only that smaller group of stemmed words and their frequency of use. Finally, I merged that dataframe with the `sentences`, `avg_sent_len`, `words`, `avg_word_len`, and `_prop` columns from earlier to create my final dataframe of features and the target variable. 

### c. Exploratory Analysis

I started off my exploratory analysis as I often do: by looking at a correlation heatmap. But this time, with over 1200 features, I narrowed it to only the features that weren't individual words. It was a bit daunting that all of the correlations were weaker than 0.1 or -0.1 except for `NOUN_prop` which had a -0.12 correlation with the r/Socialism subreddit. 

![Correlation heatmap](/charts/corr_heatmap.png)

To get an idea of how different the vocabularies were on the subreddits since it didn't seem feasible to do a heatmap, I created a venn diagram to analyze how much overlap there was between the top 100 most common words in each subreddit. This was also a bit daunting as they shared 73 of their top 100 words and only 27 words in each subreddit's top 100 words were different. 

![Common word venn diagram](/charts/top10_word_venn.png)

Then I went on to look at the mentions of the topics of each subreddit to see if they mentioned each other as much as they mentioned themselves. I found that the subreddits were significantly more likely to mention their own topic than the other topic  but still mentioned the other topic quite a bit. 

![Topic mentions](/charts/topic_mentions.png)

I also wanted to see if there was any strong disparities in the sentiment of posts in these subreddits. I found that they had largely similar average positivity, neutrality, negativity, and compound scores, but if anything, r/Capitalism was more emotional and less neutral in their posts. 

![Sentiment comparisons](/charts/sentiment_comparisons.png)

After this exploration, I was not particularly encouraged that I'd be able to predict a particularly accurate model to differentiate between these subreddits. The language and sentiment seems quite similar. But I was hoping the combination of all of these features would be more meaningful than any of them on their own.


### d. Modeling

The baseline accuracy is 0.510 for comparison. For all modeling, I used a `random_state` of 26.

I tried out models in order of increasing complexity for this project. So the first model I tried out was a LogisticRegression. Any parameters I tweaked resulted in lower scores so I used the default parameters and got a score of 0.877 on the training data and 0.726 on the testing data. This model was definitely overfit, but it was encouraging to see the simplest model outperform the baseline accuracy by about 50%. I had less success with KNeighborsClassifier, DecisionTreeClassifier, and BaggingClassifier, all of which had testing scores below 0.700.

But as I tried other ensemble models, I had increasing success. The best of these was the ExtraTrees and AdaBoost with ExtraTrees as the `base_estimator`. Then I utilized the top 5 performing models in a VotingClassifier for my best testing score of 0.773. I was able to very slightly bring that testing score up by reducing the features a bit based on the coefficients from my top performing models. When I reduced the features from 1200+ to 800, I got a final testing score of 0.774. All scores are below for your comparison. 

|Model|Parameters|Training Score|Testing Score|
|---|---|---|---|
|LogisticRegression|All default parameters|0.877|0.726|
|KNeighbors|algorithm = 'ball_tree', n_neighbors = 15, p = 1|0.684|0.615|
|DecisionTree|min_samples_leaf = 5, min_samples_split = 10, ccp_alpha = 0.001|0.892|0.663|
|Bagging|All default parameters|0.988|0.692|
|RandomForest|n_estimators = 150|0.999|0.738|
|ExtraTrees|min_samples_split = 5, n_estimators = 200, warm_start = True|0.998|0.756|
|AdaBoost|base_estimator = ExtraTreesClassifier(), learning_rate = 0.9, algorithm = 'SAMME'|0.999|0.768|
|GradientBoost|learning_rate = 0.08, max_depth = 4, n_estimators = 150|0.874|0.753|
|Voting|estimators = [LogisticRegression(), RandomForestClassifier(), ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier], weight = proportional by model testing score|0.998|0.773|
|Voting2|Same as above but with fewer features|0.998|0.774|



## IV. Conclusion

In the end, my best model had a training score of 0.998 and testing score of 0.774, so it was quite overfit but also outperformed the baseline accuracy by 51.7% on unseen data. The following features were the most signficant across the top performing models used in the final VotingClassifier.

![Strongest features](/charts/strongest_features.png)

The feature with the strongest association with r/Socialism was words with the stem 'engag', ('engaged', 'engagement', 'engaging') with 'comrad' ('comrade', 'comradery') following closely behind. Then words like 'sector', 'pass', 'leftist', and 'class' followed. Interestingly, r/Socialist seems to use more proper nouns and pronouns than r/Capitalism, which I think can explain the use of more particles too (possessive apostrophes at the end of a word may get removed so an 's' is then counted as a standalone letter for example). For r/Capitalism, fascinatingly, the number on strongest feature associated with that subreddit was use of the word 'stupid'. After that were words/stems like 'poll', 'factor', 'market', 'venezuela', and 'tax'. There seems to be more use of verbs, nouns, spaces, and possibly usernames ('latestagecapit' and 'modelusgov') among this subreddit as well.

So to conclude, it seems like there are overlapping vocabularies between r/Socialism and r/Capitalism but the way in which the language is used can help create a fairly accurate classification model. As for the problem statement, I think this could be further refined with more data and further modeling to put a finer point on the difference between the way socialists and capitalists speak. With more work, I think this type of model could be used for targeting social media users for various calls to action.

Thank you for reading all of this, dear reader! I hope it was interesting and useful for any of your future endeavors. Godspeed :^)
