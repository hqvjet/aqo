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
					 List *relids, int *fss_hash)
{
	int		nfeatures;
	double	*matrix[aqo_K];
	double	targets[aqo_K];
	double	*features;
    double  *w;
    double  *m;
    double  *v;
	double	result;
	int		rows;
	int		i;
    int     w_len;

	*fss_hash = get_fss_for_object(restrict_clauses, selectivities, relids,
														&nfeatures, &features);

    w_len = aqo_RANK * nfeatures + 1;

    w = palloc(sizeof(double) * w_len);
    m = palloc(sizeof(double) * w_len);
    v = palloc(sizeof(double) * w_len);

	if (nfeatures > 0)
		for (i = 0; i < aqo_K; ++i)
			matrix[i] = palloc0(sizeof(**matrix) * nfeatures);
    elog(WARNING, "palloc successful on cardinality_esti");

	if (load_fss(query_context.fspace_hash, *fss_hash, nfeatures, matrix,
				 targets, w, m, v, &rows))
		result = OPRr_predict(nfeatures, features, w);
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
    elog(WARNING, "predict successful on car_esti");

	pfree(features);
    pfree(w);
    pfree(m);
    pfree(v);
	if (nfeatures > 0)
	{
		for (i = 0; i < aqo_K; ++i)
			pfree(matrix[i]);
	}

	if (result < 0)
		return -1;
	else
		return clamp_row_est(exp(result));
}
