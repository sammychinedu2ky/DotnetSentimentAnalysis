using CsvHelper.Configuration;
using CsvHelper.Configuration.Attributes;
using Microsoft.ML.Data;


public class SentimentDataTransformed
{
   
    public string SentimentText { get; set; }
  
   
    public bool Sentiment { get; set; }
}



public class SentimentDataMap : ClassMap<SentimentDataTransformed>
{
    public SentimentDataMap()
    {
        Map(m => m.SentimentText).Name("review");
        Map(m => m.Sentiment)
            .Convert(args => args.Row.GetField("sentiment")
                .Equals("positive", StringComparison.OrdinalIgnoreCase));
    }
}


public class SentimentPrediction : SentimentDataTransformed
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }
}
