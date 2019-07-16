using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace InadimplentePrediction
{
    public class BeneficiarioData
    {
        //NroPlano,NroCusteio,Idade,NroConveniada,NroSituacao,NroInscricao,SeqCliente,Inadimplente
        [LoadColumn(0)]
        public float NroPlano;

        [LoadColumn(1)]
        public float NroCusteio;

        [LoadColumn(2)]
        public float Idade;

        [LoadColumn(3)]
        public float NroConveniada;

        [LoadColumn(4)]
        public float NroSituacao;

        [LoadColumn(5)]
        public float NroInscricao;

        [LoadColumn(6)]
        public float SeqCliente;

        [LoadColumn(7)]//, ColumnName("Label")]
        public float Inadimplente;
    }

    public class BeneficiarioPrediction //: BeneficiarioData
    {
        //[ColumnName("Score")]
        //public Int32 PredictedInadimplente;

        [ColumnName("PredictedLabel")] public float PredictedInadimplente;
        public float Probability;
        public float Score;
    }
}
