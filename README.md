# nyt_gender_prediction

Gendered pronouns: he, she, her, him, his, hers, (all contractions with she and he, operationalized by she and an apostrophe, until the next white space), himself, herself

Use gendered pronouns to generate ground truth labels, eliminate from dataset afterwards 

Leave named entities in because: could be an interesting proxy for professional stature (i.e. more company names in a man's review might indicate professional connectedness)

Do baseline (logistic regression with unigrams) on toy dataset 

Oracle: each of us made human predictions on 100 samples and determined oracle accuracy (f1 score) 

Evaluation: f1 score 
