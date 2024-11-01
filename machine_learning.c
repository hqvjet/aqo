/*
 *******************************************************************************
 *
 *	MACHINE LEARNING TECHNIQUES
 *
 * This module does not know anything about DBMS, cardinalities and all other
 * stuff. It learns matrices, predicts values and is quite happy.
 * The proposed method is designed for working with limited number of objects.
 * It is guaranteed that number of rows in the matrix will not exceed aqo_K
 * setting after learning procedure. This property also allows to adapt to
 * workloads which properties are slowly changed.
 *
 *******************************************************************************
 *
 * Copyright (c) 2016-2021, Postgres Professional
 *
 * IDENTIFICATION
 *	  aqo/machine_learning.c
 *
 */

#include "aqo.h"
// #include<math.h>
// #include<stdlib.h>
// #include<stdio.h>

#define B1 0.9
#define B2 0.99
#define ALPHA 1e-8
#define learning_rate 1e-2
// #define aqo_RANK 2
// #define aqo_epoch 100

static void update_weights(int nfeatures, double *w, double *m, double *v, double *x, double error, int t);

void update_weights(int nfeatures, double *w, double *m, double *v, double *x, double error, int t) {
    double g = error;
    int sub_rank;
    double m_corr, v_corr;

    m[0] = B1 * m[0] + (1 - B1) * g; 
    v[0] = B2 * v[0] + (1 - B2) * pow(g, 2);

    m_corr = m[0] / (1 - pow(B1, t));
    v_corr = v[0] / (1 - pow(B2, t));

    w[0] -= learning_rate * m_corr / (sqrt(v_corr) + ALPHA);
    
    for (int rank = 0; rank < aqo_RANK; ++ rank) {
        for (int col = 0; col < nfeatures; ++ col) {
            g = error * pow(x[col], rank + 1);
            sub_rank = nfeatures * rank + col + 1;

            m[sub_rank] = B1 * m[sub_rank] + (1 - B1) * g; 
            v[sub_rank] = B2 * v[sub_rank] + (1 - B2) * pow(g, 2);

            m_corr = m[sub_rank] / (1 - pow(B1, t));
            v_corr = v[sub_rank] / (1 - pow(B2, t));

            w[sub_rank] -= learning_rate * m_corr / (sqrt(v_corr) + ALPHA); 
        }
    }

}
/*
 * With given matrix, targets and features makes prediction for current object.
 *
 * Returns negative value in the case of refusal to make a prediction, because
 * positive targets are assumed.
 */
double
OPRr_predict(int ncols, double *features, double *w)
{
    double result = 1.0 * w[0];

    for (int rank = 0; rank < aqo_RANK; ++ rank)
        for (int col = 0; col < ncols; ++ col) {
            result += (double)(w[rank * ncols + col + 1] * pow(features[col], rank + 1));
        }

    if (result < 0)
        result = 0;

    return result;
}

/*
 * Modifies given matrix and targets using features and target value of new
            sub_rank = col;
 * object.
 * Returns indexes of changed lines: if index of line is less than matrix_rows
 * updates this line in database, otherwise adds new line with given index.
 * It is supposed that indexes of new lines are consequent numbers
 * starting from matrix_rows.
 */
int
OPRr_learn(int nrows, int nfeatures, double **matrix, double *targets, 
           double *features, double target, double *w, double *m, double *v)
{
    double error;

    for (int epoch = 1; epoch <= aqo_epoch; ++ epoch) {
        for (int row = 0; row < nrows; ++ row) {
            error = targets[row] - OPRr_predict(nfeatures, matrix[row], w);
            printf("\n");

            update_weights(nfeatures, w, m, v, matrix[row], error, epoch);
        }

        update_weights(nfeatures, w, m, v, features, target, epoch);

    }

    return nrows + 1;
}

// int main() {
//
//     printf("passed 113\n");
//     int nrows = 4;
//     int nfeatures = 7;
//     double temp_matrix[4][7] = {{-2.655181526924674e-2,379489.1310107702,1.0642580724683649e-3,-6.071141817925366e-1,-2.6551816311121274e-2,1.0642577721150653e-3,0},
//         {-2.268056795189301e+2,1.186651003306643e-2,1.8473855347065337e-9,2.282731897313745e-7,3.3250371877351972e+1,5.5840325945756756e+2,5.0979855465985e-3},
//         {2.1219957934e-3,8.487983164e-3,6.3659873764e-3,1.69759663287e-3,3e-3,0,0},
//         {2.121995792e-3,1.0609978958e-3,1.5e-3,8.487983168e-3,3.5e-3,0,0}};
//     double temp_targets[4] = {5.09807034593093e-3,6.95301158536524e-3,6.47774094685893e-3,6.95301158535496e-3};
//     double **matrix = (double**) malloc(nrows * sizeof(double*)); 
//     double *targets = (double *) malloc(nrows * sizeof(double));
//
//     for (int i = 0; i < nrows; ++i)
//         matrix[i] = (double *) malloc (nfeatures * sizeof(double));
//
//     for (int i = 0; i < nrows; ++i)
//         targets[i] = temp_targets[i];
//
//     for (int i = 0; i < nrows; ++i)
//          matrix[i] = temp_matrix[i];
//
//     double *w = (double*) malloc((nfeatures * aqo_RANK + 1) * sizeof(double));
//     double *m = (double*) malloc((nfeatures * aqo_RANK + 1) * sizeof(double));
//     double *v = (double*) malloc((nfeatures * aqo_RANK + 1) * sizeof(double));
//
//     for (int i = 0; i < nfeatures * aqo_RANK + 1; ++ i) {
//         w[i] = 0.0;
//         m[i] = 0.0;
//         v[i] = 0.0;
//     }
//
//     double *features = (double *) malloc(nfeatures * sizeof(double));
//     double temp_features[7] = {-9.47593749637022e-1,6.7987358202656836e-2,6.675606451883867e+4,6.675607111460889e+4,2.9336032677e-3,1.06099789573e-3,0};
//
//     for (int i = 0; i < nfeatures; ++i)
//         features[i] = temp_features[i];
//
//     double target = 5.098070293073e-3;
//
//     OPRr_learn(nrows, nfeatures, matrix, targets, features, target, w, m, v);
// }
