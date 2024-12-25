using Microsoft.ML;
using Microsoft.ML.Data;
using CsvHelper;
using static Microsoft.ML.DataOperationsCatalog;
using CsvHelper.Configuration;
using System.Globalization;


string _dataPath = Path.Combine(Environment.CurrentDirectory, "IMDB Dataset.csv");
//string _dataPath = "/Users/swacblooms/Documents/codes/sentimentanalysis/sentimentanalysis/IMDB Dataset.csv";
MLContext mlContext = new MLContext();

TrainTestData splitDataView = LoadData(mlContext);
ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
//Evaluate(mlContext, model, splitDataView.TestSet);

UseModelWithSingleItem(mlContext, model);
UseModelWithBatchItems(mlContext, model);
TrainTestData LoadData(MLContext mlContext)
{
    IDataView dataView;
    var config = new CsvConfiguration(CultureInfo.InvariantCulture)
    {
        PrepareHeaderForMatch = args => args.Header.ToLower(),
        HasHeaderRecord = true,
        HeaderValidated = null, 
        Delimiter = ",",
        BadDataFound = null 
    };
    using (var reader = new StreamReader(_dataPath))
    using (var csv = new CsvReader(reader, config))
    {
     csv.Context.RegisterClassMap<SentimentDataMap>();
        var records = csv.GetRecords<SentimentDataTransformed>().ToList();
        dataView = mlContext.Data.LoadFromEnumerable(records);
    }

    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    return splitDataView;
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(SentimentDataTransformed.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(SentimentDataTransformed.Sentiment),
            featureColumnName: "Features"));

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();
    return model;
}



void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView predictions = model.Transform(splitTestSet);
    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "PredictedLabel");
    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");
}

void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    PredictionEngine<SentimentDataTransformed, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentDataTransformed, SentimentPrediction>(model);
    SentimentDataTransformed sampleStatement = new SentimentDataTransformed
    {
        SentimentText = "This was a very bad steak"
    };
    var resultPrediction = predictionFunction.Predict(sampleStatement);
    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}


void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    IEnumerable<SentimentDataTransformed> sentiments = new[]
    {
    new SentimentDataTransformed
    {
        SentimentText = "This was a horrible meal"
    },
    new SentimentDataTransformed
    {
        SentimentText = "I love this spaghetti."
    }

};

    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

    IDataView predictions = model.Transform(batchComments);

    IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine();

    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
    foreach (SentimentPrediction prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }
    Console.WriteLine("=============== End of predictions ===============");
}