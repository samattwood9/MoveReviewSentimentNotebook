# Sentiment Analysis for Movie Reviews

## Installing packages and getting set-up


```C#
#r "nuget:Microsoft.ML,1.5.2"
    
using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
```


Installing package Microsoft.ML, version 1.5.2................done!



Successfully added reference to package Microsoft.ML, version 1.5.2


## Declaring data-classes


```C#
// a class for the movie reviews we're going to analyse
public class SentimentReview
{
    [LoadColumn(1)]
    public string Sentiment { get; set; }

    [LoadColumn(0)]
    public string Review { get; set; }
}
```


```C#
// a class for the predictions we're going to make
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }
}
```


```C#
// a class that will help us out later and be used to transform data so that our model can better understand it
public class LookupMap
{
    public string Value { get; set; }
    public bool Category { get; set; }
}
```

## Building the model


```C#
// create mlContext, using a seed so that results are deterministic
MLContext mlContext = new MLContext(seed: 0);
```


```C#
// load the data into an IDataView and then display its form (or schema)
string dataPath = "./imdbdataset.csv";
IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentReview>(dataPath, hasHeader: true, separatorChar: ',', allowQuoting: true);
display(dataView.Schema);
```


<table><thead><tr><th><i>index</i></th><th>Name</th><th>Index</th><th>IsHidden</th><th>Type</th><th>Annotations</th></tr></thead><tbody><tr><td>0</td><td>Sentiment</td><td>0</td><td>False</td><td>{ Microsoft.ML.Data.TextDataViewType: RawType: System.ReadOnlyMemory&lt;System.Char&gt; }</td><td>{ Microsoft.ML.DataViewSchema+Annotations: Schema: [  ] }</td></tr><tr><td>1</td><td>Review</td><td>1</td><td>False</td><td>{ Microsoft.ML.Data.TextDataViewType: RawType: System.ReadOnlyMemory&lt;System.Char&gt; }</td><td>{ Microsoft.ML.DataViewSchema+Annotations: Schema: [  ] }</td></tr></tbody></table>



```C#
// split data into training and testing sets
TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
IDataView trainingData = trainTestSplit.TrainSet;
IDataView testData = trainTestSplit.TestSet;

// and now take a quick look at both sets
display(h4("trainingData Schema"));
display(trainingData.Schema);

display(h4("testData Schema"));
display(testData.Schema);
```


<h4>trainingData Schema</h4>



<table><thead><tr><th><i>index</i></th><th>Name</th><th>Index</th><th>IsHidden</th><th>Type</th><th>Annotations</th></tr></thead><tbody><tr><td>0</td><td>Sentiment</td><td>0</td><td>False</td><td>{ Microsoft.ML.Data.TextDataViewType: RawType: System.ReadOnlyMemory&lt;System.Char&gt; }</td><td>{ Microsoft.ML.DataViewSchema+Annotations: Schema: [  ] }</td></tr><tr><td>1</td><td>Review</td><td>1</td><td>False</td><td>{ Microsoft.ML.Data.TextDataViewType: RawType: System.ReadOnlyMemory&lt;System.Char&gt; }</td><td>{ Microsoft.ML.DataViewSchema+Annotations: Schema: [  ] }</td></tr></tbody></table>



<h4>testData Schema</h4>



<table><thead><tr><th><i>index</i></th><th>Name</th><th>Index</th><th>IsHidden</th><th>Type</th><th>Annotations</th></tr></thead><tbody><tr><td>0</td><td>Sentiment</td><td>0</td><td>False</td><td>{ Microsoft.ML.Data.TextDataViewType: RawType: System.ReadOnlyMemory&lt;System.Char&gt; }</td><td>{ Microsoft.ML.DataViewSchema+Annotations: Schema: [  ] }</td></tr><tr><td>1</td><td>Review</td><td>1</td><td>False</td><td>{ Microsoft.ML.Data.TextDataViewType: RawType: System.ReadOnlyMemory&lt;System.Char&gt; }</td><td>{ Microsoft.ML.DataViewSchema+Annotations: Schema: [  ] }</td></tr></tbody></table>



```C#
// define table used to map from string values in our csv to bool values that our model can work with  
var lookupData = new[] {
    new LookupMap { Value = "negative", Category = false },
    new LookupMap { Value = "positive", Category = true }
};

var lookupIdvMap = mlContext.Data.LoadFromEnumerable(lookupData);

display(lookupIdvMap.Schema)
```


<table><thead><tr><th><i>index</i></th><th>Name</th><th>Index</th><th>IsHidden</th><th>Type</th><th>Annotations</th></tr></thead><tbody><tr><td>0</td><td>Value</td><td>0</td><td>False</td><td>{ Microsoft.ML.Data.TextDataViewType: RawType: System.ReadOnlyMemory&lt;System.Char&gt; }</td><td>{ Microsoft.ML.DataViewSchema+Annotations: Schema: [  ] }</td></tr><tr><td>1</td><td>Category</td><td>1</td><td>False</td><td>{ Microsoft.ML.Data.BooleanDataViewType: RawType: System.Boolean }</td><td>{ Microsoft.ML.DataViewSchema+Annotations: Schema: [  ] }</td></tr></tbody></table>



```C#
// make pipeline (by applying the table from the previous cell)
var dataProcessPipeline = mlContext.Transforms.Conversion.MapValue(outputColumnName: "Label", lookupMap: lookupIdvMap, lookupIdvMap.Schema["Value"], lookupIdvMap.Schema["Category"], inputColumnName: nameof(SentimentReview.Sentiment))
    .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentReview.Review)));

// set the training algorithm                         
var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");

// add the training algorithm to the pipeline
var trainingPipeline = dataProcessPipeline.Append(trainer);

display(trainingPipeline)
```


<table><thead><tr><th>LastEstimator</th></tr></thead><tbody><tr><td>{ Microsoft.ML.Trainers.SdcaLogisticRegressionBinaryTrainer: Info: { Microsoft.ML.TrainerInfo: NeedNormalization: True, WantCaching: True }, FeatureColumn: { Microsoft.ML.SchemaShape+Column: Name: Features, Kind: { Microsoft.ML.SchemaShape+Column+VectorKind: value__: 1 }, ItemType: { Microsoft.ML.Data.NumberDataViewType: RawType: System.Single }, IsKey: False, Annotations: [  ] }, LabelColumn: { Microsoft.ML.SchemaShape+Column: Name: Label, Kind: { Microsoft.ML.SchemaShape+Column+VectorKind: value__: 0 }, ItemType: { Microsoft.ML.Data.BooleanDataViewType: RawType: System.Boolean }, IsKey: False, Annotations: [  ] }, WeightColumn: { Microsoft.ML.SchemaShape+Column: Name: &lt;null&gt;, Kind: { Microsoft.ML.SchemaShape+Column+VectorKind: value__: 0 }, ItemType: &lt;null&gt;, IsKey: False, Annotations: &lt;null&gt; } }</td></tr></tbody></table>



```C#
// train the model (fitting to the trainingData)
Console.WriteLine("Please wait. The model is currently being trained (and tested)...");

ITransformer trainedModel = trainingPipeline.Fit(trainingData);

Console.WriteLine("Model trained!")
```

    Please wait. The model is currently being trained (and tested)...
    Model trained!
    


```C#
// evaluate the model on the test data
var predictions = trainedModel.Transform(testData);

var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

display(metrics)
```


<table><thead><tr><th>LogLoss</th><th>LogLossReduction</th><th>Entropy</th><th>AreaUnderRocCurve</th><th>Accuracy</th><th>PositivePrecision</th><th>PositiveRecall</th><th>NegativePrecision</th><th>NegativeRecall</th><th>F1Score</th><th>AreaUnderPrecisionRecallCurve</th><th>ConfusionMatrix</th></tr></thead><tbody><tr><td>0.5006565972153878</td><td>0.49929115971158444</td><td>0.9998956617722234</td><td>0.9292770964818572</td><td>0.8579664049299275</td><td>0.8553959627329193</td><td>0.8656452563347083</td><td>0.8606640863719699</td><td>0.8501006036217303</td><td>0.8604900907937127</td><td>0.9282273148610037</td><td>{ Microsoft.ML.Data.ConfusionMatrix: PerClassPrecision: [ 0.8553959627329193, 0.8606640863719699 ], PerClassRecall: [ 0.8656452563347083, 0.8501006036217303 ], Counts: [ [ 4407, 684 ], [ 745, 4225 ] ], NumberOfClasses: 2 }</td></tr></tbody></table>



```C#
// create a prediction engine using the trained model
var predEngine = mlContext.Model.CreatePredictionEngine<SentimentReview, SentimentPrediction>(trainedModel);

Console.WriteLine("Prediction model/engine built!")
```

    Prediction model/engine built!


## Using the model


```C#
// create some example reviews (for testing the prediction engine)
SentimentReview badReview = new SentimentReview { Review = "I hate this movie! It is terrible!" };
SentimentReview goodReview = new SentimentReview { Review = "I love this movie! It is great!" };
SentimentReview neutralReview = new SentimentReview { Review = "I don't know about this movie. It is OK." };

display(h4("Bad Review"));
display(badReview);

display(h4("Good Review"));
display(goodReview);

display(h4("Neutral Review"));
display(neutralReview);
```


<h4>Bad Review</h4>



<table><thead><tr><th>Sentiment</th><th>Review</th></tr></thead><tbody><tr><td>&lt;null&gt;</td><td>I hate this movie! It is terrible!</td></tr></tbody></table>



<h4>Good Review</h4>



<table><thead><tr><th>Sentiment</th><th>Review</th></tr></thead><tbody><tr><td>&lt;null&gt;</td><td>I love this movie! It is great!</td></tr></tbody></table>



<h4>Neutral Review</h4>



<table><thead><tr><th>Sentiment</th><th>Review</th></tr></thead><tbody><tr><td>&lt;null&gt;</td><td>I don&#39;t know about this movie. It is OK.</td></tr></tbody></table>



```C#
// predict whether each example review has a positive or negative sentiment
var predBadReview = predEngine.Predict(badReview);
var predGoodReview = predEngine.Predict(goodReview);
var predNeutralReview = predEngine.Predict(neutralReview);

display(h4("Bad Review"));
display(predBadReview);

display(h4("Good Review"));
display(predGoodReview);

display(h4("Neutral Review"));
display(predNeutralReview);
```


<h4>Bad Review</h4>



<table><thead><tr><th>Prediction</th><th>Probability</th><th>Score</th></tr></thead><tbody><tr><td>False</td><td>0.105195455</td><td>-2.1407852</td></tr></tbody></table>



<h4>Good Review</h4>



<table><thead><tr><th>Prediction</th><th>Probability</th><th>Score</th></tr></thead><tbody><tr><td>True</td><td>0.9982779</td><td>6.362488</td></tr></tbody></table>



<h4>Neutral Review</h4>



<table><thead><tr><th>Prediction</th><th>Probability</th><th>Score</th></tr></thead><tbody><tr><td>True</td><td>0.5594775</td><td>0.2390418</td></tr></tbody></table>



```C#

```
