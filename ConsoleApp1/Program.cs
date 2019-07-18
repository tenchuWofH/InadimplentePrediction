// <SnippetAddUsings>
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
// </SnippetAddUsings>
namespace InadimplentePrediction
{
    class Program
    {
        // <SnippetDeclareGlobalVariables>
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "Pesquisa_Beneficiarios_ML_20190709-train.csv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "Pesquisa_Beneficiarios_ML_20190709-test.csv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<BeneficiarioData, BeneficiarioPrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            // <SnippetCreateContext>
            _mlContext = new MLContext(seed: 1);
            // </SnippetCreateContext>

            // <SnippetCreateDataView>
            // Load Datasets
            IDataView dataView = _mlContext.Data.LoadFromTextFile<BeneficiarioData>(_trainDataPath, hasHeader: false, separatorChar: ',');
            // </SnippetCreateDataView>

            // <SnippetCreatePipeline>
            //string featuresColumnName = "Features";
            //var pipeline = _mlContext.Transforms
            //    .Concatenate(featuresColumnName, "NroPlano", "NroCusteio", "Idade", "NroConveniada", "NroSituacao", "NroInscricao", "SeqCliente", "Label")
            //    //.Append(_mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3)
            //    .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression())
            //    .Append(_mlContext.Transforms.CustomMapping<FromLabel, ToLabel>(
            //        mapAction: (input, output) => { output.Label = input.Label == 1 ? true : false; },
            //        contractName: null))
            //    .AppendCacheCheckpoint(_mlContext);

            ////Get all the feature column names (All except the Label and the IdPreservationColumn)
            //string[] featureColumnNames = dataView.Schema.AsQueryable()
            //    .Select(column => column.Name)                               // Get alll the column names
            //    .Where(name => name != nameof(BeneficiarioData.Inadimplente)) // Do not include the Label column
            //    //.Where(name => name != "IdPreservationColumn")               // Do not include the IdPreservationColumn/StratificationColumn
            //    //.Where(name => name != "Time")                               // Do not include the Time column. Not needed as feature column
            //    .ToArray();

            //var dataProcessPipeLine = _mlContext.Transforms.Text
            //    .FeaturizeText("Features", "Label");// nameof(BeneficiarioData.Inadimplente));
            string[] featureColumnNames = dataView.Schema.AsQueryable()
                .Select(column => column.Name)                               // Get alll the column names
                .Where(name => name != "Label") // Do not include the Label column
                .ToArray();

            var dataProcessPipeLine = _mlContext.Transforms.Concatenate("Features", featureColumnNames)
                .Append(_mlContext.Transforms.Conversion.ConvertType("Label", "Inadimplente", DataKind.Boolean));

            var trainingPipeLine = dataProcessPipeLine
                .Append(_mlContext.BinaryClassification.Trainers.FastTree());

            var cvResults = _mlContext.BinaryClassification
                .CrossValidate(dataView, estimator: trainingPipeLine);

            var accuracy = cvResults.Select(r => r.Metrics.Accuracy);
            var areaUnderRocCurve = cvResults.Select(r => r.Metrics.AreaUnderRocCurve);
            var areaUnderPrecisionRecallCurve = cvResults.Select(r => r.Metrics.AreaUnderPrecisionRecallCurve);
            var entropy = cvResults.Select(r => r.Metrics.Entropy);
            var f1Score = cvResults.Select(r => r.Metrics.F1Score);
            Console.WriteLine($"accuracy: {accuracy.Average()}");
            Console.WriteLine($"areaUnderRocCurve: {areaUnderRocCurve.Average()}");
            Console.WriteLine($"areaUnderPrecisionRecallCurve: {areaUnderPrecisionRecallCurve.Average()}");
            Console.WriteLine($"entropy: {entropy.Average()}");
            Console.WriteLine($"f1Score: {f1Score.Average()}");

            // </SnippetCreatePipeline>

            // <SnippetTrainModel>
            var model = trainingPipeLine.Fit(dataView);
            // </SnippetTrainModel>

            // <SnippetSaveModel>
            ////using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            ////{
            ////    _mlContext.Model.Save(model, dataView.Schema, fileStream);
            ////}
            //SaveModelAsFile(_mlContext, dataView.Schema, model);
            // </SnippetSaveModel>

            // <SnippetPredictor>
            var predictor = _mlContext.Model.CreatePredictionEngine<BeneficiarioData, BeneficiarioPrediction>(model);
            // </SnippetPredictor>


            // <SnippetPredictionExample>
            BeneficiarioData beneficiarioPrediction = new BeneficiarioData()
            {
                NroPlano = 9,
                NroCusteio = 0,
                Idade = 42,
                NroConveniada = 9,
                NroSituacao = 1,
                NroInscricao = 300835,
                SeqCliente = 0,
                Inadimplente = 0
            };
            
            var prediction = predictor.Predict(beneficiarioPrediction);
            Console.WriteLine($"PredictedInadimplente: {prediction.PredictedInadimplente}");
            Console.WriteLine($"Score: {string.Join(" ", prediction.Score)}");
            // </SnippetPredictionExample>
        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            // <SnippetSaveModel> 
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
            // </SnippetSaveModel>

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

    }
}