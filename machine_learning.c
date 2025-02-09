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
// #include<stdip.h>

#define MAX_DEPTH 8
#define MIN_NODE 2
#define NUM_EPOCH 10
#define LAMBDA 0.01
// #define aqo_k 3
// #define aqo_K 30
// #define learning_rate 1e-1
// #define object_selection_threshold 0.1

struct RT_Node
{
    double threshold;
    double weight[2];
    int col;
    struct RT_Node	*left;
    struct RT_Node	*right;
};

static double fs_distance(double *a, double *b, int len);
static double fs_similarity(double dist);
static double compute_weights(double *distances, int nrows, double *w, int *idx);
static void ridge_regression(int start, int end, int col, int *matrix_index, double **matrix, const double *targets, double *w);
static double calculate_error(int start, int end, int col, int *matrix_index, double **matrix, const double *targets, double *region_value);
static void calculate_loss(int nrows, int target_col, int *matrix_index, double **matrix, const double *targets, double *position, double *loss);
static struct RT_Node* create_node(double threshold, double *w, int col);
static struct RT_Node* build_tree(int nrows, int ncols, int depth, int *matrix_index, double **matrix, const double *targets, double *features);

struct RT_Node*
create_node(double threshold, double *w, int col)
{
    struct RT_Node *node = palloc(sizeof(*node));
    node->threshold = threshold;
    if (w != NULL)
    {
        node->weight[0] = w[0];
        node->weight[1] = w[1];
    }
    node->col = col;
    node->left = NULL;
    node->right = NULL;
    return node;
}

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

/*
 * Returns similarity between objects based on distance between them.
 */
double
fs_similarity(double dist)
{
	return 1.0 / (0.001 + dist);
}

/*
 * Compute weights necessary for both prediction and learning.
 * Creates and returns w, w_sum and idx based on given distances ad matrix_rows.
 *
 * Appeared as a separate function because of "don't repeat your code"
 * principle.
 */
double
compute_weights(double *distances, int nrows, double *w, int *idx)
{
	int		i,
			j;
	int		to_insert,
			tmp;
	double	w_sum = 0;

	for (i = 0; i < aqo_k; ++i)
		idx[i] = -1;

	/* Choose from all neighbors only several nearest objects */
	for (i = 0; i < nrows; ++i)
		for (j = 0; j < aqo_k; ++j)
			if (idx[j] == -1 || distances[i] < distances[idx[j]])
			{
				to_insert = i;
				for (; j < aqo_k; ++j)
				{
					tmp = idx[j];
					idx[j] = to_insert;
					to_insert = tmp;
				}
				break;
			}

	/* Compute weights by the nearest neighbors distances */
	for (j = 0; j < aqo_k && idx[j] != -1; ++j)
	{
		w[j] = fs_similarity(distances[idx[j]]);
		w_sum += w[j];
	}
	return w_sum;
}

/*
 * Theoretic approach: Ridge regression
 * y^ = w_0 + w_1*x
 * loss = 1/2 * (y^ - y)^2 + lambda * w_1^2 (Squared error)
 */
void 
ridge_regression(int start, int end, int col, int *matrix_index, double **matrix, const double *targets, double *w) {
    int i,j;

    for (i = 0; i < 20; ++i)
    {
        double gradient[2] = {0, 0};
        for (j = start; j <= end; ++j)
        {
            if (matrix_index[j] == 1)
            {
                gradient[0] += w[0] + w[1] * matrix[j][col] - targets[j];
                gradient[1] += matrix[j][col] * (w[0] + w[1] * matrix[j][col] - targets[j]) + 2 * 0.01 * w[1];
                // gradient[1] += matrix[j][col] * (w[0] + w[1] * targets[j]);
            }
        }

        // printf("w0: %.10f, w1: %.10f\n", w[0], w[1]);
        w[0] = w[0] - 0.1 * gradient[0];
        w[1] = w[1] - 0.1 * gradient[1];
    }
}

// Using Average in Region
double
calculate_error(int start, int end, int col, int *matrix_index, double **matrix, const double *targets, double *region_value) {
    double predict;
    double loss = 0;
    double w[2] = {0.1, 0.1};
    int i;

    ridge_regression(start, end, col, matrix_index, matrix, targets, w);

    region_value[0] = w[0];
    region_value[1] = w[1];

    // Calculate error based on regions value
    for (i = start; i <= end; ++i)
        if (matrix_index[i] == 1)
        {
            predict = w[0] + w[1] * matrix[i][col];
            // printf("Predict: %.2f, target: %.2f\n", predict, targets[i]);
            loss += (targets[i] - predict) * (targets[i] - predict);
        }

    return loss;
}

void
calculate_loss(int nrows, int target_col, int *matrix_index,
               double **matrix, const double *targets, double *position, double *loss)
{
    int i, j;
    double region_loss;
    double best_loss=10e9;
    double best_position = 0.0;
    double region_value[2];
    double next_feature = 0.0;

    for (i = 0; i < aqo_K - 1; ++i) {
        if (matrix_index[i] == 1)
        {
            region_loss = calculate_error(0, i, target_col, matrix_index, matrix, targets, region_value);
            region_loss += calculate_error(i + 1, aqo_K - 1, target_col, matrix_index, matrix, targets, region_value);

            // printf("Region loss: %.2f\n", region_loss);

            if (region_loss < best_loss) {
                best_loss = region_loss;
                for (j = i + 1; j < aqo_K; ++j)
                    if (matrix_index[j] == 1) {
                        next_feature = matrix[j][target_col];        
                        break;
                    }

                best_position = (matrix[i][target_col] + next_feature) / 2.0;
            }
        }
    }

    *position = best_position;
    *loss = best_loss;
}

struct RT_Node*
build_tree(int nrows, int ncols, int depth, int *matrix_index, double **matrix, const double *targets, double *features)
{
    int i;
    double  best_loss = 10e9;
    double  best_position = 0.0;
    double  loss;
    double  position;
    int     i_dimension = 0;
    int     right_matrix_index[aqo_K];
    int     left_matrix_index[aqo_K];
    int     right_nrows = nrows;
    int     left_nrows = nrows;
    struct RT_Node *root = NULL;

    // for (i = 0; i < depth - 1; ++i)
    //     printf("    ");

    for (i = 0; i < aqo_K; ++i) {
        right_matrix_index[i] = matrix_index[i];
        left_matrix_index[i] = matrix_index[i];
    }

    // Optimize Tree to find best threshold (lowest loss)
    for (i = 0; i < ncols; ++i) {
        // Search for best hyperparameters
        calculate_loss(nrows, i, matrix_index, matrix, targets, &position, &loss);
        // printf("Split: %.2f\t", position);
        if (loss < best_loss) {
            best_loss = loss;
            best_position = position;
            i_dimension = i;
        }
    }

    /* Filter matrix with the new condition */
    // Leftside
    for (i = 0; i < aqo_K; ++i)
    {
        if (left_matrix_index[i] == 1 && matrix[i][i_dimension] >= best_position)
        {    
            left_matrix_index[i] = 0;
            --left_nrows;
        }
    }

    // Rightside
    for (i = 0; i < aqo_K; ++i)
    {
        if (right_matrix_index[i] == 1 && matrix[i][i_dimension] < best_position)
        {
            right_matrix_index[i] = 0;
            --right_nrows;
        }
    }

    // Stop condition
    if (right_nrows < MIN_NODE || left_nrows < MIN_NODE || depth > MAX_DEPTH) {
        double w[2] = {0.1, 0.1};
        ridge_regression(0, aqo_K - 1, i_dimension, matrix_index, matrix, targets, w);

        // printf("Leaft: w0: %.10f, w1: %.10f\n", w[0], w[1]);

        // printf("w0: %.10f, w1: %.10f\n", w[0], w[1]);
        return create_node(NAN, w, i_dimension);
    }

    /* Filter object by new threshold */
    root = create_node(best_position, NULL, i_dimension);
    // printf("Split: %.2f, Loss: %.2f, Col: %d\n", best_position, best_loss, i_dimension);

    root->left = build_tree(left_nrows, ncols, depth + 1, left_matrix_index, matrix, targets, features);
    root->right = build_tree(right_nrows, ncols, depth + 1, right_matrix_index, matrix, targets, features);

    return root;
}

double
OkNNr_predict(int nrows, int ncols, double **matrix, const double *targets,
			  double *features)
{
	int		i;
	double	result = 0;
    int     matrix_index[aqo_K];
    int     matrix_size = nrows;
    struct RT_Node *tree = NULL;

 	/* this should never happen */
    if (nrows == 0)
		return -1;

    for (i = 0; i < aqo_K; ++i) {
        if (i < nrows)
            matrix_index[i] = 1;
        else
            matrix_index[i] = 0;
    }

    tree = build_tree(matrix_size, ncols, 1, matrix_index, matrix, targets, features);

    while (tree != NULL)
    {
        if (isnan(tree->threshold)) {
            result = tree->weight[0] + tree->weight[1] * features[tree->col];
            break;
        }
        else
        {
            if (features[tree->col] >= tree->threshold)
                tree = tree->right;
            else
                tree = tree->left;
        }
    }

	if (result < 0)
		result = 0;

	return result;
}

/*
 * Modifies given matrix and targets using features and target value of new
 * object.
 * Returns indexes of changed lines: if index of line is less than matrix_rows
 * updates this line in database, otherwise adds new line with given index.
 * It is supposed that indexes of new lines are consequent numbers
 * starting from matrix_rows.
 */
int
OkNNr_learn(int nrows, int nfeatures, double **matrix, double *targets,
			double *features, double target)
{
	double	   distances[aqo_K];
	int			i,
				j;
	int			mid = 0; /* index of row with minimum distance value */
	int		   idx[aqo_K];

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

	if (nrows < aqo_K)
	{
		/* We can't reached limit of stored neighbors */

		/*
		 * Add new line into the matrix. We can do this because matrix_rows
		 * is not the boundary of matrix. Matrix has aqo_K free lines
		 */
		for (j = 0; j < nfeatures; ++j)
			matrix[nrows][j] = features[j];
		targets[nrows] = target;

		return nrows+1;
	}
	else
	{
		double	*feature;
		double	avg_target = 0;
		double	tc_coef; /* Target correction coefficient */
		double	fc_coef; /* Feature correction coefficient */
		double	w[aqo_K];
		double	w_sum;

		/*
		 * We reaches limit of stored neighbors and can't simply add new line
		 * at the matrix. Also, we can't simply delete one of the stored
		 * neighbors.
		 */

		/*
		 * Select nearest neighbors for the new object. store its indexes in
		 * idx array. Compute weight for each nearest neighbor and total weight
		 * of all nearest neighbor.
		 */
		w_sum = compute_weights(distances, nrows, w, idx);

		/*
		 * Compute average value for target by nearest neighbors. We need to
		 * check idx[i] != -1 because we may have smaller value of nearest
		 * neighbors than aqo_k.
		 * Semantics of coef1: it is defined distance between new object and
		 * this superposition value (with linear smoothing).
		 * */
		for (i = 0; i < aqo_k && idx[i] != -1; ++i)
			avg_target += targets[idx[i]] * w[i] / w_sum;
		tc_coef = learning_rate * (avg_target - target);

		/* Modify targets and features of each nearest neighbor row. */
		for (i = 0; i < aqo_k && idx[i] != -1; ++i)
		{
			fc_coef = tc_coef * (targets[idx[i]] - avg_target) * w[i] * w[i] /
				sqrt(nfeatures) / w_sum;

			targets[idx[i]] -= tc_coef * w[i] / w_sum;
			for (j = 0; j < nfeatures; ++j)
			{
				feature = matrix[idx[i]];
				feature[j] -= fc_coef * (features[j] - feature[j]) /
					distances[idx[i]];
			}
		}
	}

	return nrows;
}

// int main() {
//     int nrows = 30;
//     int nfeatures = 3;
//     double temp_matrix[32][3] = {
//         {0.01,0.02,0.03},
//         {0.02,0.03,0.04},
//         {0.03,0.04,0.05},
//         {0.04,0.05,0.06},
//         {0.05,0.06,0.07},
//         {0.06,0.07,0.08},
//         {0.07,0.08,0.09},
//         {0.08,0.09,0.10},
//         {0.09,0.10,0.11},
//         {0.10,0.11,0.12},
//         {0.11,0.12,0.13},
//         {0.12,0.13,0.14},
//         {0.13,0.14,0.15},
//         {0.14,0.15,0.16},
//         {0.15,0.16,0.17},
//         {0.16,0.17,0.18},
//         {0.17,0.18,0.19},
//         {0.18,0.19,0.20},
//         {0.19,0.20,0.21},
//         {0.20,0.21,0.22},
//         {0.21,0.22,0.23},
//         {0.22,0.23,0.24},
//         {0.23,0.24,0.25},
//         {0.24,0.25,0.26},
//         {0.25,0.26,0.27},
//         {0.26,0.27,0.28},
//         {0.27,0.28,0.29},
//         {0.28,0.29,0.30},
//         {0.29,0.30,0.31},
//         {0.30,0.31,0.32},
//     };
//     double temp_targets[32] = {0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.24,0.27,0.30,0.33,0.36,0.39,0.42,0.45,0.48,0.51,0.54,0.57,0.60,0.63,0.66,0.69,0.72,0.75,0.78,0.81,0.84,0.87,0.90,0.93,0.96};
//
//     double **matrix = (double**) malloc(aqo_K * sizeof(double*)); 
//     double *targets = (double *) malloc(aqo_K * sizeof(double));
//
//     for (int i = 0; i < aqo_K; ++i)
//         matrix[i] = (double *) malloc (nfeatures * sizeof(double));
//
//
//     for (int i = 0; i < nrows; ++i)
//         targets[i] = temp_targets[i];
//
//     for (int i = 0; i < nrows; ++i)
//          matrix[i] = temp_matrix[i];
//
//     double *features = (double *) malloc(nfeatures * sizeof(double));
//     double temp_features[3] = {0.31,0.32,0.33};
//
//     for (int i = 0; i < nfeatures; ++i)
//         features[i] = temp_features[i];
//
//     double target = 0.99;
//     int a = OkNNr_learn(nrows, nfeatures, matrix, targets, features, target);
//
//
//     // Print matrix and targets
//     // for (int i = 0; i < nrows; ++i)
//     // {
//     //     for (int j = 0; j < nfeatures; ++j)
//     //         printf("%f ", matrix[i][j]);
//     //     printf(" %f\n", targets[i]);
//     // }
//
//     printf("Prediction: %f\n", OkNNr_predict(nrows, nfeatures, matrix, targets, features));
// }
