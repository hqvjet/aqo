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
static double evaluate(int rank, int nfeatures, double *B_matrix, double *features, double target);
static double calculate_l2_regularization(int nfeatures, double **X_matrix);

/*
 * Computes L2 to X matrix
 */
double
calculate_l2_regularization(int limit, double **matrix) {
    double l2 = 0.0;

    /*
     * Calculate L2 by 1% of averaging diagonal values of X matrix
     */
    for (int i = 0; i < limit; ++i) {
        l2 += matrix[i][i];
    }

    l2 = 0.01 * (l2 / limit);

    return l2;
}

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

/*
 * Computes inverse bias weight of model
 * We focus on using Cholesky Decomposition for solving inverse matrix
 */
int 
calculate_inverse_matrix(int nfeatures, double **X_matrix, double **X_inverse) {
    int limit = nfeatures * aqo_RANK + 1;
    // double l2 = calculate_l2_regularization(limit, X_matrix);
    double l2 = 1e-10;

    /*
     * Let X = L.L^(-1), we calculate L matrix by below code
     */
    double **L = (double **) palloc(sizeof(double*) * limit);
    double **inv_L = (double **) palloc(sizeof(double*) * limit);

    for (int i = 0; i < limit; ++i)
        L[i] = (double *) palloc(sizeof(double) * limit);
    for (int i = 0; i < limit; ++i)
        inv_L[i] = (double *) palloc(sizeof(double) * limit);

    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }

            if (i == j) {
                int L_sqrt = X_matrix[i][j] - sum;
                if (i != 0)
                    L_sqrt += l2;
                /*
                 * if matrix diagonal is not sastified positive definition, then return false
                 */
                if (L_sqrt <= EPSILON)
                    return 0;

                L[i][j] = sqrt(L_sqrt);
            } else {
                L[i][j] = (X_matrix[i][j] - sum) / L[j][j];
            }
        }
    }

    /*
     * After that, We will find L^(-1)
     */

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
            if (isinf(inv_L[i][j]))
                elog(ERROR, "Overflow occurred!\n");
        }
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

    return 1;
}

/*
 * Use for predicting cardinality
 */
double
OPRr_predict(int rank, int ncols, double *features, double *bias)
{
    double result = bias[0];
    elog(WARNING, "Rank: %d", rank);

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

    elog(WARNING, "First ele: %f", X_matrix[0][0]);

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

        elog(WARNING, "Matrix is inversible!");
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
    else {
        best_rank = 0;
        elog(WARNING, "Matrix is singular!");
    }

    pfree(X_inverse);

    return best_rank;
}
