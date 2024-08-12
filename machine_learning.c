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
#define EPSILON 1e-15

static void update_X_matrix(int nfeatures, double **matrix, double *features);
static void update_Y_matrix(int nfeatures, double *matrix, double *features, double target);
static void update_B_matrix(int nfeatures, double **X_inverse, double *Y_matrix, double *B_matrix);
static int calculate_inverse_matrix(int nfeatures, double **X_matrix, double **X_inverse);
static void swapRows(double **A, int row1, int row2, int limit);
static double evaluate(int rank, int nfeatures, double *B_matrix, double *features, double target);

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

void 
swapRows(double **A, int row1, int row2, int limit) {
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
int 
calculate_inverse_matrix(int nfeatures, double **X_matrix, double **X_inverse) {
    int limit = nfeatures * aqo_RANK + 1;
    double **temp_matrix;

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
    temp_matrix = (double **) palloc(limit * sizeof(double *));
    for (int i = 0; i < limit; i++) {
        temp_matrix[i] = (double *) palloc(limit * sizeof(double));
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
        double pivotValue;
        for (int j = i + 1; j < limit; j++) {
            if (fabs(temp_matrix[j][i]) > fabs(temp_matrix[pivot][i])) {
                pivot = j;
            }
        }

        /*
         * Check for if matrix is invertible
         */
        if (fabs(temp_matrix[pivot][i]) < EPSILON) {
            for (int i = 0; i < limit; i++) {
                pfree(temp_matrix[i]);
            }
            return 0; 
        }

        if (pivot != i) {
            swapRows(temp_matrix, i, pivot, limit);
            swapRows(X_inverse, i, pivot, limit);
        }

        pivotValue = temp_matrix[i][i];
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

    return 1;
}

double
OPRr_predict(int rank, int ncols, double *features, double *bias)
{
    double result = bias[0];

    if (rank > 0) {
        /*
         * Calculate output value based on bias weight of model
         */
        for (int i = 0; i < rank; ++i) {
            for (int j = 0; j < ncols; ++j) 
                result += bias[i*rank + j + 1] * pow(features[j], i + 1);
        }
    }

    else {
        result = -1;
    }

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
    double **X_inverse = (double **) palloc(sizeof(double *) * limit);
    double eval_scores[3];
    int best_rank = 1;

    for (int i = 0; i < limit; ++i)
        X_inverse[i] = (double *) palloc((limit) * sizeof(double));

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

        for (int i = 1; i < aqo_RANK; ++i)
            if (eval_scores[best_rank] < eval_scores[i])
                best_rank = i;
    }
    else
        best_rank = 0;

    pfree(X_inverse);

    return best_rank;
}
