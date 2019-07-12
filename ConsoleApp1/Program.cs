using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.OneHotEncodingEstimator;

namespace ConsoleApp1
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Pesquisa_Beneficiarios_ML_20190709-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Pesquisa_Beneficiarios_ML_20190709-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath);

            try
            {
                Evaluate(mlContext, model);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            try
            {
                TestSinglePrediction(mlContext, model);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            Console.ReadLine();
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<Beneficiario>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Inadimplente")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SglSexoEncoded", inputColumnName: "SglSexo"))
                //.Append(mlContext.Transforms.Conversion.ConvertType(outputColumnName: "SglSexoEncoded", inputColumnName: "SglSexo", DataKind.Int32))
                .Append(mlContext.Transforms.Concatenate("Features", "NroPlano", "NroVinculo", "SglSexoEncoded", "NroCusteio", "Idade"))
                //.Append(mlContext.Regression.Trainers.FastTree());
                .Append(mlContext.BinaryClassification.Trainers.FastTree());

            //// set up a training pipeline
            //// step 1: concatenate all feature columns
            //var pipeline = mlContext.Transforms.Concatenate(
            //    "Features",
            //    "NroPlano",
            //    "NroVinculo",
            //    "SglSexo",
            //    "NroCusteio",
            //    "Idade"
            //    )
            //// step 2: set up a fast tree learner
            //.Append(mlContext.BinaryClassification.Trainers.FastTree(
            //    labelColumnName: Beneficiario.Label,
            //    featureColumnName: DefaultColumnNames.Features));

            var model = pipeline.Fit(dataView);

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<Beneficiario>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            //var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            //Console.WriteLine();
            //Console.WriteLine($"*************************************************");
            //Console.WriteLine($"*       Model quality metrics evaluation         ");
            //Console.WriteLine($"*------------------------------------------------");
            //Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            //Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
            //Console.WriteLine($"*************************************************");

            // compare the predictions with the ground truth
            var metrics = mlContext.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: "Label",
                scoreColumnName: "Score");

            // report the results
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy:P2}");
            //Console.WriteLine($"  Auc:               {metrics.Auc:P2}");
            //Console.WriteLine($"  Auprc:             {metrics.Auprc:P2}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score:P2}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss:0.##}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction:0.##}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall:0.##}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision:0.##}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall:0.##}");
            Console.WriteLine();
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<Beneficiario, BeneficiarioPrediction>(model);

            //Sample: 
            //vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            //VTS,1,1,1140,3.75,CRD,15.5
            var beneficiarioSample = new Beneficiario()
            {
                //VendorId = "VTS",
                //RateCode = "1",
                //PassengerCount = 1,
                //TripTime = 1140,
                //TripDistance = 3.75f,
                //PaymentType = "CRD",
                //FareAmount = 0 // To predict. Actual/Observed = 15.5
                NroPlano = 9,
                NroVinculo = 0,
                SglSexo = 1,//"M",
                NroCusteio = 0,
                Idade = 42
            };

            var prediction = predictionFunction.Predict(beneficiarioSample);

            //Console.WriteLine($"**********************************************************************");
            //Console.WriteLine($"Predicted Inadimplente: {prediction.PredictedInadimplente:0.####}, actual Inadimplente: 15.5");
            //Console.WriteLine($"**********************************************************************");

            // report the results
            Console.WriteLine($"  NroPlano: {beneficiarioSample.NroPlano} ");
            Console.WriteLine($"  NroVinculo: {beneficiarioSample.NroVinculo} ");
            Console.WriteLine($"  SglSexo: {beneficiarioSample.SglSexo} ");
            Console.WriteLine($"  NroCusteio: {beneficiarioSample.NroCusteio} ");
            Console.WriteLine($"  Idade: {beneficiarioSample.Idade} ");
            Console.WriteLine();
            Console.WriteLine($"Prediction: {(prediction.PredictedInadimplente ? "É Inadimplente" : "Não é Inadimplente")} ");
            Console.WriteLine($"Probability: {prediction.Probability} ");
        }
    }
}
