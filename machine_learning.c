/*
 *******************************************************************************
 *
 *	MACHINE LEARNING TECHNIQUES
 *
 * This module does not know anything about DBMS, cardinalities and all other
 * stuff. It learns matrices, predicts values and is quite happy.
 *
 *******************************************************************************
 *
 * Copyright (c) 2024, Hoang Quoc Viet
 *
 * IDENTIFICATION
 *	  aqo/machine_learning.c
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include "aqo.h"
#define aqo_RANK (3)

/*
 * Computes each weights of X matrix
 * The X matrix has shape like reverse stairs, because of its symmetrical at the diagonal of the square
 */
void
update_X_matrix(int nfeatures, double **matrix, double *features) {
    int limit = nfeatures * aqo_RANK + 1;
    int index_i = 0, index_j = 0, pow_i = 0, pow_j = 0;

    /*
     * Update weights for (limit x limit) matrix, we calculate the position under consideration
     * by search for its index and exponential level
     */
    for (int i=0; i<limit; ++i) {
        for (int j=0; j<(limit-i); ++j) {
            if (i != 0 || j != 0) {
                index_i = (i - 1) % nfeatures;
                pow_i = ceil((i * 1.0) / nfeatures);
                /*
                 * Because of symmetrical, we must plus i to keep all j position in order
                 */
                index_j = (j + i - 1) % nfeatures;
                pow_j = ceil((j + i * 1.0) / nfeatures);

                matrix[i][j] += pow(features[index_i], pow_i) * pow(features[index_j], pow_j);
                printf("i = %d, j = %d, x_%d^%d*x_%d^%d = %f\n", i, j, index_i + 1, pow_i, index_j + 1, pow_j, matrix[i][j]);
            }

            else 
                matrix[i][j] += 1;
        }
    }
}

void
update_Y_matrix(int nfeatures, double **matrix, double *features, double target) {
    
}

void
update_B_matrix(int nfeatures, double **X_matrix, double *Y_matrix, double *B_matrix) {

}

void calculate_inverse_matrix(int nfeatures, double **X_matrix, double **X_inverse) {

}

double 
evaluate(int nfeatures, double *B_matrix, double *features, double target) {

}

int
OPRr_learn(double **X_matrix, double *Y_matrix, double *B_matrix, int nfeatures,
                        double *features, double target)
{
    // double eval_scores[3];
    //
    // for (int rank=1; rank<=max_rank; ++rank) {
    //     /*
    //      * We plus new features into these matrices for learning
    //      */
    //     update_X_matrix(nfeatures, X_matrix, features);
    //     update_Y_matrix(nfeatures, Y_matrix, features, target);
    //
    //     /*
    //      * We find the inverse matrix and then, we will get new bias weights using B = X^(-1).Y
    //      * Finally, We will calculate evaluation of model with equation <rank>
    //      */
    //     double **X_inverse = calculate_inverse_matrix(nfeatures, X_matrix);
    //     update_B_matrix(nfeatures, X_matrix, Y_matrix, B_matrix);
    //     eval_score[rank-1] = evaluate(nfeatures, B_matrix, features, target);
    // }
    //
    // /*
    //  * We search for the lowest loss model based on new features, target
    //  * Our weekness is the loss model is only based on one new features. target. That may lead to
    //  * a bit partial to new features
    //  */
    // int best_rank = 1;
    //
    // for (int i=2; i<=max_rank; ++i)
    //     if (eval_scores[best_rank] < eval_scores[i])
    //         best_rank = i;
    //
    // return best_rank;
    
    update_X_matrix(nfeatures, X_matrix, features);

    int limit = nfeatures*aqo_RANK + 1;
    for(int i=0; i<limit; ++i) {
        for (int j=0; j<limit-i; ++j)
            printf("%f ", X_matrix[i][j]);
        printf("\n");
    }

    return 1;
}

double
OPRr_predict(int rank, int ncols, double *features, double *bias)
{
    double result = bias[0];

    /*
     * Calculate output value based on bias weight of model
     */
    for (int i=0; i<rank; ++i)
        for (int j=0; j<ncols; ++j) 
            result += bias[i*rank + j + 1] * pow(features[j], i + 1);

	return result;
}

int main() {
    int nfeatures = 3;
    int rank = 2;
    int limit = nfeatures * aqo_RANK + 1;
    double target = 2.83333;
    double **X_matrix = (double **)malloc(limit * sizeof(double *));
    for (int i=0; i<limit; ++i)
        X_matrix[i] = (double *)malloc((limit - i) * sizeof(double));

    for (int i = 0; i < limit; ++i)
        for (int j=0; j <limit-i; ++j)
            X_matrix[i][j] = 0.0;

    double *Y_matrix = (double *)malloc(limit * sizeof(double));
    double *B_matrix = (double *)malloc(limit * sizeof(double));

    double *features = (double *)malloc(nfeatures * sizeof(double));
    for (int i=0; i<nfeatures; ++i)
        features[i] = 3.0;

    int a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
}
