using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ConsoleApp1
{
    public class Beneficiario
    {
        [LoadColumn(0)]
        public float NroPlano;

        [LoadColumn(1)]
        public float NroVinculo;

        [LoadColumn(2)]
        public float SglSexo;

        [LoadColumn(3)]
        public float NroCusteio;

        [LoadColumn(4)]
        public float Idade;

        [LoadColumn(5)]//, ColumnName("Label")]
        public bool Inadimplente;
    }

    public class BeneficiarioPrediction
    {
        //[ColumnName("Score")]
        //public Int32 PredictedInadimplente;

        [ColumnName("PredictedLabel")] public bool PredictedInadimplente;
        public float Probability;
        public float Score;
    }
}
