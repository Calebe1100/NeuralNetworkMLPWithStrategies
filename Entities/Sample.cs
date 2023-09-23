﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkMLP.Entities
{
    public class Sample
    {
        public double[] CordX { get; set; }
        public double[] CordY { get; set; }

        public Sample(int lengthX, int lengthY)
        {
            CordX = new double[lengthX];
            CordY = new double[lengthY];
        }
    }
}
