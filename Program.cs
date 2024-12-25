using Microsoft.ML;
using Microsoft.ML.Data;
using CsvHelper;
using static Microsoft.ML.DataOperationsCatalog;
using CsvHelper.Configuration;
using System.Globalization;


string _dataPath = Path.Combine(Environment.CurrentDirectory, "IMDB Dataset.csv");
string modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model.zip");
MLContext mlContext = new MLContext();

ITransformer model;

if (File.Exists(modelPath))
{
    model = LoadTrainedModel(mlContext, out var modelSchema);
}
else
{
    TrainTestData splitDataView = LoadData(mlContext);
    model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
    Evaluate(mlContext, model, splitDataView.TestSet);
}


TestModelWithInput(mlContext, model);

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

ITransformer LoadTrainedModel(MLContext mlContext, out DataViewSchema modelSchema)
{
    string modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model.zip");
    if (!File.Exists(modelPath))
    {
        throw new FileNotFoundException($"Model file not found: {modelPath}");
    }
    Console.WriteLine($"Loading model from {modelPath}");
    ITransformer trainedModel = mlContext.Model.Load(modelPath, out modelSchema);
    return trainedModel;
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
    mlContext.Model.Save(model, splitTrainSet.Schema, modelPath);
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

void TestModelWithInput(MLContext mlContext, ITransformer model)
{
    PredictionEngine<SentimentDataTransformed, SentimentPrediction> predictionFunction =
        mlContext.Model.CreatePredictionEngine<SentimentDataTransformed, SentimentPrediction>(model);

    Console.WriteLine("Enter a sentence to test sentiment (press Enter to use the default: 'This was a very bad steak'):");
    string? inputText = Console.ReadLine();

    if (string.IsNullOrWhiteSpace(inputText))
    {
        inputText = "This was a very bad steak";
    }

    SentimentDataTransformed sampleStatement = new SentimentDataTransformed
    {
        SentimentText = inputText
    };

    var resultPrediction = predictionFunction.Predict(sampleStatement);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");
    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}
