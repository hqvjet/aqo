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

#include "aqo.h"
#define aqo_RANK (3)
#define EPSILON 1e-15

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
                matrix[i][j] += 1.0;
        }
    }

    // printf
//     printf("X matrix:\n");
//     for (int i = 0; i < limit; ++i) {
//         for (int j = 0; j < limit; ++j)
//             printf("%f ", matrix[i][j]);
//         printf("\n");
//     }
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
 * We focus on using Gauss Jordan Elimination for solving inverse matrix
 */
int calculate_inverse_matrix(int nfeatures, double **X_matrix, double **X_inverse) {
    int limit = nfeatures * aqo_RANK + 1;

    /*
     * Init inverse matrix
     */
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < limit; ++j) {
            X_inverse[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    /*
     * Copy data from X matrix
     */
    double **temp_matrix = (double **)malloc(limit * sizeof(double *));
    for (int i = 0; i < limit; i++) {
        temp_matrix[i] = (double *)malloc(limit * sizeof(double));
    }

    for (int i = 0; i < limit; i++) {
        for (int j = 0; j < limit; j++) {
            temp_matrix[i][j] = X_matrix[i][j];
        }
    }

    /*
     * Gauss Jordan Elimination
     */
    for (int i = 0; i < limit; i++) {
        int pivot = i;
        for (int j = i + 1; j < limit; j++) {
            if (fabs(temp_matrix[j][i]) > fabs(temp_matrix[pivot][i])) {
                pivot = j;
            }
        }

        /*
         * Check for if matrix is invertible
         */
        if (fabs(temp_matrix[pivot][i]) < EPSILON) {
            return 0; 
        }

        if (pivot != i) {
            swapRows(temp_matrix, i, pivot, limit);
            swapRows(X_inverse, i, pivot, limit);
        }

        double pivotValue = temp_matrix[i][i];
        for (int j = 0; j < limit; ++j) {
            temp_matrix[i][j] /= pivotValue;
            X_inverse[i][j] /= pivotValue;
        }

        for (int k = 0; k < limit; ++k) {
            if (k != i) {
                double factor = temp_matrix[k][i];
                for (int j = 0; j < limit; j++) {
                    temp_matrix[k][j] -= factor * temp_matrix[i][j];
                    X_inverse[k][j] -= factor * X_inverse[i][j];
                }
            }
        }
    }

    // printf("X inverse:\n");
    // for (int i = 0; i < limit; ++i) {
    //     for (int j = 0; j < limit; ++j)
    //         printf("%f ", X_inverse[i][j]);
    //     printf("\n");
    // }
    //
    // double I[limit][limit];
    //
    // for (int i = 0; i < limit; ++i) {
    //     for (int j = 0; j <limit; ++j) {
    //         I[i][j] = 0;
    //     }
    // }
    //
    // for (int i = 0; i < limit; ++i) {
    //     for (int j = 0; j <limit; ++j) {
    //         for (int k = 0; k < limit; ++k)
    //             I[i][j] += X_matrix[i][k] * X_inverse[k][j];
    //     }
    // }
    //
    // printf("I matrix:\n");
    // for (int i = 0; i < limit; ++i) {
    //     for (int j = 0; j < limit; ++j)
    //         printf("%f ", I[i][j]);
    //     printf("\n");
    // }
    // printf("Temp matrix:\n");
    // for (int i = 0; i < limit; ++i) {
    //     for (int j = 0; j < limit; ++j)
    //         printf("%f ", temp_matrix[i][j]);
    //     printf("\n");
    // }

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

/*
 * evaluate the model
 */
double 
evaluate(int rank, int nfeatures, double *B_matrix, double *features, double target) {
    double y_pred = OPRr_predict(rank, nfeatures, features, B_matrix);
    return fabs(target - y_pred);
}

int
OPRr_learn(double **X_matrix, double *Y_matrix, double *B_matrix, int nfeatures,
                        double *features, double target)
{
    int limit = nfeatures * aqo_RANK + 1;
    double **X_inverse = (double **)malloc(limit * sizeof(double *));
    for (int i = 0; i < limit; ++i)
        X_inverse[i] = (double *)malloc((limit) * sizeof(double));
    double eval_scores[3];

    /*
     * We plus new features into these matrices for learning
     */
    update_X_matrix(nfeatures, X_matrix, features);
    update_Y_matrix(nfeatures, Y_matrix, features, target);

    /*
     * We find the inverse matrix and then, we will get new bias weights using B = X^(-1).Y
     * Finally, We will calculate evaluation of model with equation <rank>
     * If the matrix is not inversible, then we will skip bias updating and use old bias
     * Later, we will use pseudo inverse for non-inversible matrix
     */
    if (calculate_inverse_matrix(nfeatures, X_matrix, X_inverse)) {
        update_B_matrix(nfeatures, X_matrix, Y_matrix, B_matrix);

        for (int rank = 1; rank <= aqo_RANK; ++rank) 
            eval_scores[rank-1] = evaluate(rank, nfeatures, B_matrix, features, target);

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
    }
    else
        return 0;
}

// int main() {
//     int nfeatures = 3;
//     int rank = 2;
//     int limit = nfeatures * aqo_RANK + 1;
//     double target;
//     double **X_matrix = (double **)malloc(limit * sizeof(double *));
//     for (int i=0; i<limit; ++i)
//         X_matrix[i] = (double *)malloc(limit * sizeof(double));
//
//     for (int i = 0; i < limit; ++i)
//         for (int j=0; j <= i; ++j)
//             X_matrix[i][j] = 0.0;
//
//     double *Y_matrix = (double *)malloc(limit * sizeof(double));
//     for (int i=0; i<limit; ++i)
//         Y_matrix[i] = 0.0;
//
//     double *B_matrix = (double *)malloc(limit * sizeof(double));
//
//     double *features = (double *)malloc(nfeatures * sizeof(double));
//
//     features[0] = 0.123450;
//     features[1] = 0.223450;
//     features[2] = 0.323450;
//     target = 2.8883333;
//     int a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//
//     features[0] = 0.5488;
//     features[1] = 0.7152;
//     features[2] = 0.6028;
//     target = 2.8883333;
//     a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//
//     features[0] = 0.4435;
//     features[1] = 0.00443;
//     features[2] = 0.23;
//     target = 1.3234;
//     a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//
//     features[0] = 0.88;
//     features[1] = 0.52;
//     features[2] = 0.28;
//     target = 3.21;
//     a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//
//     features[0] = 0.43823;
//     features[1] = 0.344524;
//     features[2] = 0.48742;
//     target = 1.56;
//     a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//
//     features[0] = 0.3564273;
//     features[1] = 0.9843;
//     features[2] = 0.124;
//     target = 3.91;
//     a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//
//     features[0] = 0.42345;
//     features[1] = 0.434441;
//     features[2] = 0.200231;
//     target = 1.21;
//     a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//
//     features[0] = 0.31123213;
//     features[1] = 0.00023842;
//     features[2] = 0.11;
//     target = 2.22;
//     a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//
//     features[0] = 0.32411;
//     features[1] = 0.66453;
//     features[2] = 0.12434;
//     target = 2.3234;
//     a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//
//     features[0] = 0.23213;
//     features[1] = 0.0123842;
//     features[2] = 0.34411123;
//     target = 2.123;
//     a = OPRr_learn(X_matrix, Y_matrix, B_matrix, nfeatures, features, target);
//     printf("Best rank: %d\n", a);
// }
