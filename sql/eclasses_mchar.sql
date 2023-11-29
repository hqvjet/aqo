-- Testing for working with equivalence classes for mchar type

-- Skip test if mchar extension does not exist
SELECT count(*) = 0 AS skip_test
FROM pg_available_extensions WHERE name = 'mchar' \gset

\if :skip_test
\quit
\endif

CREATE EXTENSION IF NOT EXISTS aqo;
SET aqo.show_details = 'on';
SET aqo.show_hash = 'off';
SET aqo.mode = 'forced';

-- MCHAR fields
CREATE EXTENSION MCHAR;
CREATE TABLE aqo_test_mchar(a mchar, b mchar, c mchar);
INSERT INTO aqo_test_mchar
SELECT (x/10)::text::mchar, (x/100)::text::mchar, (x/1000)::text::mchar
FROM generate_series(0, 9999) x;
ANALYZE aqo_test_mchar;

SELECT true AS success FROM aqo_reset();
-- Not equivalent queries
EXPLAIN (ANALYZE, COSTS OFF, SUMMARY OFF, TIMING OFF)
SELECT * FROM aqo_test_mchar
WHERE a = b AND a = '0';

EXPLAIN (ANALYZE, COSTS OFF, SUMMARY OFF, TIMING OFF)
SELECT * FROM aqo_test_mchar
WHERE a = c AND a = '0';

EXPLAIN (ANALYZE, COSTS OFF, SUMMARY OFF, TIMING OFF)
SELECT * FROM aqo_test_mchar
WHERE b = c AND a = '0';

-- Must be 3
SELECT count(*) FROM aqo_data;
SELECT true AS success FROM aqo_reset();

-- Equivalent queries
EXPLAIN (ANALYZE, COSTS OFF, SUMMARY OFF, TIMING OFF)
SELECT * FROM aqo_test_mchar
WHERE a = b AND a = c AND a = '0';

EXPLAIN (ANALYZE, COSTS OFF, SUMMARY OFF, TIMING OFF)
SELECT * FROM aqo_test_mchar
WHERE a = b AND b = c AND a = '0';

EXPLAIN (ANALYZE, COSTS OFF, SUMMARY OFF, TIMING OFF)
SELECT * FROM aqo_test_mchar
WHERE a = b AND a = c AND b = c AND a = '0';

EXPLAIN (ANALYZE, COSTS OFF, SUMMARY OFF, TIMING OFF)
SELECT * FROM aqo_test_mchar
WHERE a = b AND b = c AND a = '0' AND b = '0';

EXPLAIN (ANALYZE, COSTS OFF, SUMMARY OFF, TIMING OFF)
SELECT * FROM aqo_test_mchar
WHERE a = b AND b = c AND a = '0' AND c = '0';

EXPLAIN (ANALYZE, COSTS OFF, SUMMARY OFF, TIMING OFF)
SELECT * FROM aqo_test_mchar
WHERE a = b AND b = c AND a = '0' AND b = '0' AND c = '0';
-- Must be 1
SELECT count(*) FROM aqo_data;
SELECT true AS success FROM aqo_reset();

DROP TABLE aqo_test_mchar;

DROP EXTENSION mchar;
DROP EXTENSION aqo;
