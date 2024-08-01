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
#include <string.h>
// #include "aqo.h"
#define aqo_RANK (3)
#define EPSILON 1e-10

/*
 * Computes each weights of X matrix
 */
void
update_X_matrix(int nfeatures, double **matrix, double *features) {
    int limit = nfeatures * aqo_RANK + 1;
    int index_i, index_j, pow_i, pow_j;

    /*
     * Update weights for (limit x limit) matrix, we calculate the position under consideration
     * by search for its index and exponential level
     */
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < limit; ++j) {
            if (i != 0 || j != 0) {
                index_i = (i - 1) % nfeatures;
                pow_i = ceil((i * 1.0) / nfeatures);
                index_j = (j - 1) % nfeatures;
                pow_j = ceil((j * 1.0) / nfeatures);

                matrix[i][j] += pow(features[index_i], pow_i) * pow(features[index_j], pow_j);
            }

            else 
                matrix[i][j] += 1;
        }
    }
}

/*
 * Computes each weights of Y matrix
 */
void
update_Y_matrix(int nfeatures, double *matrix, double *features, double target) {
    int limit = nfeatures * aqo_RANK + 1;
    int index_i, pow_i;

    for (int i = 0; i < limit; ++i) {
        index_i = (i - 1) % nfeatures;
        pow_i = ceil((i * 1.0) / nfeatures);

        matrix[i] += target * pow(features[index_i], pow_i);
    }
}

/*
 * Computes each weights of B matrix
 */
void
update_B_matrix(int nfeatures, double **X_inverse, double *Y_matrix, double *B_matrix) {
    int limit = nfeatures * aqo_RANK + 1;
    for (int i = 0; i < limit; ++i) 
        for (int j = 0; j < limit; ++j) 
            B_matrix[i] += X_inverse[i][j] * Y_matrix[j];
}

void swapRows(double **A, int row1, int row2, int limit) {
    for (int i = 0; i < limit; ++i) {
        double temp = A[row1][i];
        A[row1][i] = A[row2][i];
        A[row2][i] = temp;
    }
}

/*
 * Computes inverse bias weight of model
 * We focus on using modified Cholesky Decomposition for solving inverse matrix
 */
int calculate_inverse_matrix(int nfeatures, double **X_matrix, double **X_inverse) {
    int limit = nfeatures * aqo_RANK + 1;

    /*
     * Let X = L.L^(-1), we calculate L matrix by below code
     * For avoid non-positive definition matrix, we replace B[i][j] = 0 by epsilon
     */

    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < limit; ++j) {
            X_inverse[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    /*
     * Gauss Jordan Elimination
     */
    for (int i = 0; i < limit; i++) {
        int pivot = i;
        for (int j = i + 1; j < limit; j++) {
            if (fabs(X_matrix[j][i]) > fabs(X_matrix[pivot][i])) {
                pivot = j;
            }
        }
        printf("%f ", X_matrix[pivot][i]);

        /*
         * Check for if matrix is invertible
         */
        if (fabs(X_matrix[pivot][i]) < EPSILON) {
            return 0; 
        }

        if (pivot != i) {
            swapRows(X_matrix, i, pivot, limit);
            swapRows(X_inverse, i, pivot, limit);
        }

        double pivotValue = X_matrix[i][i];
        for (int j = 0; j < limit; ++j) {
            X_matrix[i][j] /= pivotValue;
            X_inverse[i][j] /= pivotValue;
        }

        for (int k = 0; k < limit; ++k) {
            if (k != i) {
                double factor = X_matrix[k][i];
                for (int j = 0; j < limit; j++) {
                    X_matrix[k][j] -= factor * X_matrix[i][j];
                    X_inverse[k][j] -= factor * X_inverse[i][j];
                }
            }
        }
    }

    return 1;
}

double 
evaluate(int nfeatures, double *B_matrix, double *features, double target) {
    return 0.0;
}

int
OPRr_learn(double **X_matrix, double *Y_matrix, double *B_matrix, int nfeatures,
                        double *features, double target)
{
    int limit = nfeatures * aqo_RANK + 1;
    double **X_inverse = (double **)malloc(limit * sizeof(double *));
    for (int i=0; i<limit; ++i)
        X_inverse[i] = (double *)malloc((limit) * sizeof(double));
    double eval_scores[3];

    for (int rank = 1; rank <= aqo_RANK; ++rank) {
        /*
         * We plus new features into these matrices for learning
         */
        update_X_matrix(nfeatures, X_matrix, features);
        update_Y_matrix(nfeatures, Y_matrix, features, target);

        /*
         * We find the inverse matrix and then, we will get new bias weights using B = X^(-1).Y
         * Finally, We will calculate evaluation of model with equation <rank>
         * If the matrix is not inversible, then we will skip bias updating and use old bias
         */
        if (calculate_inverse_matrix(nfeatures, X_matrix, X_inverse)) {
            update_B_matrix(nfeatures, X_matrix, Y_matrix, B_matrix);
            eval_scores[rank-1] = evaluate(nfeatures, B_matrix, features, target);
        }
    }

    /*
     * We search for the lowest loss model based on new features, target
     * Our weekness is the loss model is only based on one new features. target. That may lead to
     * a bit partial to new features
     */
    int best_rank = 1;

    for (int i=2; i <= aqo_RANK; ++i)
        if (eval_scores[best_rank] < eval_scores[i])
            best_rank = i;

    return best_rank;
    
    // update_X_matrix(nfeatures, X_matrix, features);
    //
    // printf("X matrix:\n");
    // for(int i=0; i<limit; i++) {
    //     for (int j=0; j < limit; j++)
    //         printf("%f ", X_matrix[i][j]);
    //     printf("\n");
    // }
    //
    // update_Y_matrix(nfeatures, Y_matrix, features, target);
    //
    // int is_invertible = calculate_inverse_matrix(nfeatures, X_matrix, X_inverse);
    //
    // if (is_invertible == 0)
    //     printf("X matrix is not invertible");
    //
    // else {
    //     printf("inverse X matrix:\n");
    //     for(int i=0; i<limit; ++i) {
    //         for (int j=0; j<limit; ++j)
    //             printf("%f ", X_inverse[i][j]);
    //         printf("\n");
    //     }
    //
    //     update_B_matrix(nfeatures, X_inverse, Y_matrix, B_matrix);
    //
    //     printf("B matrix:\n");
    //     for(int i=0; i<limit; ++i) {
    //         printf("%f\n", B_matrix[i]);
    //     }
    // }
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
        X_matrix[i] = (double *)malloc(limit * sizeof(double));

    for (int i = 0; i < limit; ++i)
        for (int j=0; j <= i; ++j)
            X_matrix[i][j] = 0.0;

    double *Y_matrix = (double *)malloc(limit * sizeof(double));
    for (int i=0; i<limit; ++i)
        Y_matrix[i] = 0.0;

    double *B_matrix = (double *)malloc(limit * sizeof(double));

    double *features = (double *)malloc(nfeatures * sizeof(double));
    features[0] = 0.5488;
    features[1] = 0.7152;
    features[2] = 0.6028;

    int a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
}
