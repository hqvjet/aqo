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

/*
 * Computes each weights of X matrix
 * The X matrix has shape like reverse stairs, because of its symmetrical at the diagonal of the square
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
        for (int j = 0; j <= i; ++j) {
            if (i != 0 || j != 0) {
                index_i = (i - 1) % nfeatures;
                pow_i = ceil((i * 1.0) / nfeatures);
                index_j = (j - 1) % nfeatures;
                pow_j = ceil((j * 1.0) / nfeatures);

                matrix[i][j] += pow(features[index_i], pow_i) * pow(features[index_j], pow_j);
                // printf("i = %d, j = %d, x_%d^%d*x_%d^%d = %f\n", i, j, index_i + 1, pow_i, index_j + 1, pow_j, matrix[i][j]);
                // printf("%f, %f\n", features[index_i], features[index_j]);
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

void
update_B_matrix(int nfeatures, double **X_matrix, double *Y_matrix, double *B_matrix) {


}

/*
 * Computes inverse bias weight of model
 * We focus on using Cholesky Decomposition for solving inverse matrix
 */
void calculate_inverse_matrix(int nfeatures, double **X_matrix, double **X_inverse) {
    int limit = nfeatures * aqo_RANK + 1;

    /*
     * Let X = L.L^(-1), we calculate L matrix by below code
     */
    double L[limit][limit], epsilon = 1e-13; 
    int idx_1, idx_2;
    memset(L, 0, sizeof(L));

    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (j > i) {
                idx_1 = j;
                idx_2 = i;
            }
            else {
                idx_1 = i;
                idx_2 = j;
            }
            if (i == j) {
                L[i][j] = fmax(sqrt(X_matrix[idx_1][idx_2] - sum), epsilon);
            } else {
                L[i][j] = fmax((X_matrix[idx_1][idx_2] - sum) / L[j][j], epsilon);
            }
        }
    }

    /*
     * After that, We will find L^(-1)
     */
    double inv_L[limit][limit];
    memset(inv_L, 0, sizeof(inv_L));

    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j <= i; ++j) {
            double minus = 0.0;
            for (int k = j; k < limit - 1; ++k) {
                minus -= L[i][k] * inv_L[k][j];
            }
            if (i == j) {
                inv_L[i][j] = 1.0 / L[i][j];
            }
            else {
                inv_L[i][j] = (minus * 1.0) / L[i][i]; 
            }
            printf("inv_L[%d][%d] = %f ", i, j, inv_L[i][j]);
            printf("minus: %f\n", minus);
        }
    }
    printf("L matrix:\n");
    for(int i=0; i<limit; ++i) {
        for (int j=0; j<limit; ++j)
            printf("%f ", L[i][j]);
        printf("\n");
    }
    printf("inverse L matrix:\n");
    for(int i=0; i<limit; ++i) {
        for (int j=0; j<limit; ++j)
            printf("%f ", inv_L[i][j]);
        printf("\n");
    }

    /*
     * Finally, We have L and L^(-1), now we will calculate inverse matrix by
     * X = L.L^T <=> X^(-1) = (L^T)^(-1).L^(-1)
     */

    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < limit; ++j) {
            for (int k = 0; k < limit; ++k) {
                X_inverse[i][j] += inv_L[k][i] * inv_L[k][j]; 
            }
        }
    }
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

    int limit = nfeatures * aqo_RANK + 1;
    double **X_inverse = (double **)malloc(limit * sizeof(double *));
    for (int i=0; i<limit; ++i)
        X_inverse[i] = (double *)malloc((limit) * sizeof(double));

    for (int i=0; i<limit; ++i)
        for (int j=0; j<limit; ++j)
            X_inverse[i][j] = 0.0;
    
    update_X_matrix(nfeatures, X_matrix, features);

    printf("X matrix:\n");
    for(int i=0; i<limit; i++) {
        for (int j=0; j < limit; j++)
            if (j > i)
                printf("%f ", X_matrix[j][i]);
            else
                printf("%f ", X_matrix[i][j]);
        printf("\n");
    }

    update_Y_matrix(nfeatures, Y_matrix, features, target);

    printf("Y matrix:\n");
    for(int i=0; i<limit; ++i) {
        printf("%f ", Y_matrix[i]);
        printf("\n");
    }

    // double **matrix = (double **)malloc(3 * sizeof(double *));
    // for (int i=0; i<3; ++i)
    //     matrix[i] = (double *)malloc((i+1) * sizeof(double));
    //
    // matrix[0][0] = 1;
    // matrix[1][0] = 2;
    // matrix[1][1] = 5;
    // matrix[2][0] = 4;
    // matrix[2][1] = 10;
    // matrix[2][2] = 21;

    calculate_inverse_matrix(nfeatures, X_matrix, X_inverse);

    printf("inverse X matrix:\n");
    for(int i=0; i<limit; ++i) {
        for (int j=0; j<limit; ++j)
            printf("%f ", X_inverse[i][j]);
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
        X_matrix[i] = (double *)malloc((i + 1) * sizeof(double));

    for (int i = 0; i < limit; ++i)
        for (int j=0; j <= i; ++j)
            X_matrix[i][j] = 0.0;

    double *Y_matrix = (double *)malloc(limit * sizeof(double));
    for (int i=0; i<limit; ++i)
        Y_matrix[i] = 0.0;

    double *B_matrix = (double *)malloc(limit * sizeof(double));

    double *features = (double *)malloc(nfeatures * sizeof(double));
    for (int i=0; i<nfeatures; ++i)
        features[i] = i + 1.0;

    int a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
}
