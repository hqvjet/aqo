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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LEN 100

/*
 *  Main idea: y_pred = b0 + b1 * x_1,0 + b2 * x_2,0 + b3 * x_1,1 + b4 * x_2,1 + .... 
 *  Max equation rank: 3
 */

static void compute_X_B(double (*X)[3], double *B, double **matrix, double *targets, int nrows, int ncols);
static double compute_det3x3(double **matrix);
static void compute_inverse_matrix(double (*X)[3], double **matrix);
static void multiply_matrix(double (*X)[3], double *BIAS, double *bias);
static void get_bias(double *bias);
static void save_bias(double *bias);
static double fs_distance(double *a, double *b, int len);

/*
 * Computes L2-distance between two given vectors.
 */
double
fs_distance(double *a, double *b, int len)
{
	double		res = 0;
	int			i;

	for (i = 0; i < len; ++i)
		res += (a[i] - b[i]) * (a[i] - b[i]);
	if (len != 0)
		res = sqrt(res / len);
	return res;
}

void
get_bias(double *bias) {
    FILE *file;
    char line[MAX_LINE_LEN];
    double bias_0, bias_1, bias_2;

    file = fopen("aqo.conf", "r");
    if (file == NULL) {
        elog(ERROR, "Can not read aqo.conf");
    }
    else {
        while (fgets(line, sizeof(line), file)) {
            if (line[0] == '#') {
                continue;
            }

            if (strstr(line, "bias_0") != NULL) {
                sscanf(line, "bias_0 = %lf", &bias_0);
            } else if (strstr(line, "bias_1") != NULL) {
                sscanf(line, "bias_1 = %lf", &bias_1);
            } else if (strstr(line, "bias_2") != NULL) {
                sscanf(line, "bias_2 = %lf", &bias_2);
            }
        }

        fclose(file);

        bias[0] = bias_0;
        bias[1] = bias_1;
        bias[2] = bias_2;
    }
}

void
save_bias(double *bias) {
    FILE *file;

    file = fopen("aqo.conf", "w");
    if (file == NULL) {
        elog(ERROR, "Can not read aqo.conf");
    }
    else {
        fprintf(file, "# aqo.conf - AQO configuration file\n");
        fprintf(file, "# Polynomial Regression model parameters\n");
        fprintf(file, "bias_0 = %.3f\n", bias[0]);
        fprintf(file, "bias_1 = %.3f\n", bias[1]);
        fprintf(file, "bias_2 = %.3f\n", bias[2]);

        fclose(file);
    }
}

/*
*   Compute X
*/
void
compute_X_B(double (*X)[3], double *B, double **matrix, double *targets, int nrows, int ncols) {

    double  sum_x = 0,
            sum_y = 0,
            sum_xy = 0,
            sum_x2 = 0,
            sum_x2y = 0,
            sum_x3 = 0,
            sum_x4 = 0;

    /* Calc element of X */
    for (int i=0; i<nrows; ++i) {
        double  temp_x = 0;

        sum_y += targets[i];

        for (int j=0; j<ncols; ++j) {
            sum_x += matrix[i][j];
            temp_x += matrix[i][j];
        }

        sum_x2 += pow(temp_x, 2);
        sum_x3 += pow(temp_x, 3);
        sum_x4 += pow(temp_x, 4);

        sum_xy += sum_x * sum_y;
        sum_x2y += sum_x2 * sum_y;
    }

    X[0][0] = nrows;
    X[0][1] = sum_x;
    X[0][2] = sum_x2;
    X[1][0] = X[0][1];
    X[1][1] = X[0][2];
    X[1][2] = sum_x3;
    X[2][0] = X[1][1];
    X[2][1] = X[1][2];
    X[2][2] = sum_x4;

    B[0] = sum_y;
    B[1] = sum_xy;
    B[2] = sum_x2y;
}

double 
compute_det3x3(double **matrix) {
    return    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[2][0] * matrix[1][2])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[2][0] * matrix[1][1]);
}

void 
compute_inverse_matrix(double (*X)[3], double **matrix) {
    double det = compute_det3x3(matrix);

    if (det == 1e-10) {
        elog(ERROR, "Determinant equal 0");
    }
    
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            X[i][j] *= 1 / det;
        }
    }
}

void 
multiply_matrix(double (*X)[3], double *BIAS, double *bias) {
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            bias[i] += X[i][j]*BIAS[j];
        }
    }
}

double
OPRr_predict(int nrows, double *features)
{
    double b[3];
    double sum_x = 0;
    double result = 0;

    get_bias(b);

    for(int i=0; i<nrows; ++i)
        sum_x += features[i];

    result = b[0] + b[1] * sum_x + b[2] * sum_x*sum_x;

	return result;
}

int
OPRr_learn(int nrows, int nfeatures, double **matrix, double *targets,
			double *features, double target)
{
	double	   distances[aqo_K];
	int			i,
				j;
	int			mid = 0; /* index of row with minimum distance value */
    double X[3][3], B[3], b[3];

	/*
	 * For each neighbor compute distance and search for nearest object.
	 */
	for (i = 0; i < nrows; ++i)
	{
		distances[i] = fs_distance(matrix[i], features, nfeatures);
		if (distances[i] < distances[mid])
			mid = i;
	}

	/*
	 * We do not want to add new very similar neighbor. And we can't
	 * replace data for the neighbor to avoid some fluctuations.
	 * We will change it's row with linear smoothing by learning_rate.
	 */
	if (nrows > 0 && distances[mid] < object_selection_threshold)
	{
		for (j = 0; j < nfeatures; ++j)
			matrix[mid][j] += learning_rate * (features[j] - matrix[mid][j]);
		targets[mid] += learning_rate * (target - targets[mid]);

		return nrows;
	}

    /* Init X, B, b */
    for (int i=0; i<3; ++i){
        B[i]=0, b[i]=0;
        for (int j=0; j<3; ++j)
            X[i][j]=0;
    }

    compute_X_B(X, B, matrix, targets, nrows, nfeatures);
    compute_inverse_matrix(X, matrix);
    multiply_matrix(X, B, b);
    save_bias(b);
   
    return nrows;
}
