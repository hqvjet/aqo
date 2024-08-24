/*
 *******************************************************************************
 *
 *	CARDINALITY ESTIMATION
 *
 * This is the module in which cardinality estimation problem obtained from
 * cardinality_hooks turns into machine learning problem.
 *
 *******************************************************************************
 *
 * Copyright (c) 2016-2021, Postgres Professional
 *
 * IDENTIFICATION
 *	  aqo/cardinality_estimation.c
 *
 */

#include "aqo.h"
#include "optimizer/optimizer.h"

/*
 * General method for prediction the cardinality of given relation.
 */
double
predict_for_relation(List *restrict_clauses, List *selectivities,
					 List *relids, int *fss_hash, double *feature_num)
{
	int		nfeatures;
	double	*features;
    int     rank;
    int     limit;
	double	result;
    int		i;
    double	**X_matrix;
	double	*Y_matrix;
    double  *B_matrix;

	*fss_hash = get_fss_for_object(restrict_clauses, selectivities, relids,
														&nfeatures, &features);
    limit = nfeatures * aqo_RANK + 1;
    X_matrix = (double **) palloc(sizeof(double *) * limit);
	Y_matrix = (double *) palloc(sizeof(double) * limit);
    B_matrix = (double *) palloc(sizeof(double) * limit);

	if (nfeatures > 0)
		for (i = 0; i < limit; ++i)
			X_matrix[i] = palloc(sizeof(**X_matrix) * limit);

	if (load_fss(query_context.fspace_hash, *fss_hash, &rank, nfeatures, X_matrix,
				 Y_matrix, B_matrix)) {
        feature_num = &X_matrix[0][0];
		result = OPRr_predict(rank, nfeatures, features, B_matrix);
    }
	else
	{
		/*
		 * Due to planning optimizer tries to build many alternate paths. Many
		 * of these not used in final query execution path. Consequently, only
		 * small part of paths was used for AQO learning and fetch into the AQO
		 * knowledge base.
		 */
		result = -1;
	}

	pfree(features);
    pfree(B_matrix);
    pfree(Y_matrix);
	if (nfeatures > 0)
	{
		for (i = 0; i < limit; ++i)
			pfree(X_matrix[i]);
	}

	if (result < 0)
		return -1;
	else
		return clamp_row_est(exp(result));
}
