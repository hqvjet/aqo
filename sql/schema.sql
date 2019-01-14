DROP EXTENSION IF EXISTS aqo CASCADE;
DROP SCHEMA IF EXISTS test CASCADE;

-- Check Zero-schema path behaviour
CREATE SCHEMA IF NOT EXISTS test;
SET search_path TO test;
DROP SCHEMA IF EXISTS test CASCADE;
CREATE EXTENSION aqo;  -- fail

-- Check default schema switching after AQO initialization
CREATE SCHEMA IF NOT EXISTS test1;
SET search_path TO test1, public;
CREATE EXTENSION aqo;
SET aqo.mode = 'intelligent';

CREATE TABLE test (id SERIAL, data TEXT);
INSERT INTO test (data) VALUES ('string');

SELECT * FROM test;
SELECT * FROM public.aqo_query_texts;
SELECT * FROM public.aqo_queries;
DROP SCHEMA IF EXISTS test1 CASCADE;
