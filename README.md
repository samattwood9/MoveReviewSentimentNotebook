# :clapper: Sentiment Analysis for Movie Reviews

*This repository demonstrates [ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) being applied to a binary classification problem: predicting the sentiment (positive or negative) of movie reviews. Go [here](MovieReviewSentimentNotebook.md) to view the notebook itself.*

| ML.NET version | API type          | App Type    | Data type | Scenario            | ML Task                   | Algorithm                  |
|----------------|-------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.5.2         | Dynamic API       | Jupyter Notebook | CSV       | Sentiment Analysis  | Two-class  classification | Stochastic dual coordinate ascent |

## Getting Started

1) Get the data! It's available on kaggle here: [imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (and used and described in [this paper](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+Word+Vectors+for+Sentiment+Analysis&btnG=).)

2) Fork or download this repository.

3) Put the CSV file, downloaded in step 1, inside the repository.

4) Change the name of the CSV file to: *imdbdataset.csv* (or change the code so that it works with the original file name).

```
MovieReviewSentimentNotebook
└─── imdbdataset.csv
└─── MovieReviewSentimentNotebook.ipynb
└─── ...
```

5) Open up the notebook!

## Additional Information

Another version of this work, in the form of a console app instead of a notebook, can be found here: https://github.com/samattwood9/MovieReviewSentiment
