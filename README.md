# Adaptive query optimization

Adaptive query optimization is the extension of standard PostgreSQL cost-based
query optimizer. Its basic principle is to use query execution statistics
for improving cardinality estimation. Experimental evaluation shows that this
improvement sometimes provides an enormously large speed-up for rather
complicated queries.

## Info
I replace 3NN online by Mutiple Polynomial Regression online.
This repo is now currently usable.
