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
        public float NroPlano { get; set; }

        [LoadColumn(1)]
        public float NroCusteio { get; set; }

        [LoadColumn(2)]
        public float Idade { get; set; }

        [LoadColumn(3)]
        public float NroConveniada { get; set; }

        [LoadColumn(4)]
        public float NroSituacao { get; set; }

        [LoadColumn(5)]
        public float NroInscricao { get; set; }

        [LoadColumn(6)]
        public float SeqCliente { get; set; }

        [LoadColumn(7)]//, ColumnName("Label")]
        public float Inadimplente { get; set; }
        //[LoadColumn(7)]
        //public string Label;
    }

    public class BeneficiarioPrediction: BeneficiarioData
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedInadimplente { get; set; }

        //public float Probability { get; set; }
        public float Score { get; set; }
    }

    ///// <summary>
    ///// This class describes which input columns we want to transform.
    ///// </summary>
    //public class FromLabel
    //{
    //    public float Label { get; set; }
    //}

    ///// <summary>
    ///// This class describes what output columns we want to produce.
    ///// </summary>
    //public class ToLabel
    //{
    //    public bool Label { get; set; }
    //}
}
