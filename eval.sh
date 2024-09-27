FILES=(	"1a.sql" "1b.sql" "1c.sql" "1d.sql" 
	"2a.sql" "2b.sql" "2c.sql" "2d.sql"
	"3a.sql" "3b.sql" "3c.sql"
	"4a.sql" "4b.sql" "4c.sql"
	"5a.sql" "5b.sql" "5c.sql"
	"6a.sql" "6b.sql" "6c.sql" "6d.sql" "6e.sql" "6f.sql"
	"7a.sql" "7b.sql" "7c.sql"
	"8a.sql" "8b.sql" "8c.sql" "8d.sql"
	"9a.sql" "9b.sql" "9c.sql" "9d.sql"
	"10a.sql" "10b.sql" "10c.sql"
	"11a.sql" "11b.sql" "11c.sql" "11d.sql"
	"12a.sql" "12b.sql" "12c.sql"
	"13a.sql" "13b.sql" "13c.sql" "13d.sql"
	"14a.sql" "14b.sql" "14c.sql"
	"15a.sql" "15b.sql" "15c.sql" "15d.sql"
	"16a.sql" "16b.sql" "16c.sql" "16d.sql"
	"17a.sql" "17b.sql" "17c.sql" "17d.sql" "17e.sql" "17f.sql"
	"18a.sql" "18b.sql" "18c.sql"
	"19a.sql" "19b.sql" "19c.sql" "19d.sql"
	"20a.sql" "20b.sql" "20c.sql"
	"21a.sql" "21b.sql" "21c.sql"
	"22a.sql" "22b.sql" "22c.sql" "22d.sql"
	"23a.sql" "23b.sql" "23c.sql"
	"24a.sql" "24b.sql"
	"25a.sql" "25b.sql" "25c.sql"
	"26a.sql" "26b.sql" "26c.sql"
	"27a.sql" "27b.sql" "27c.sql"
	"28a.sql" "28b.sql" "28c.sql"
	"29a.sql" "29b.sql" "29c.sql"
	"30a.sql" "30b.sql" "30c.sql"
	"31a.sql" "31b.sql" "31c.sql"
	"32a.sql" "32b.sql"
	"33a.sql" "33b.sql" "33c.sql"
	)

REPEAT=60

/usr/local/pgsql/bin/psql -U postgres -d test_knn -c "set aqo.mode = 'learn';"
/usr/local/pgsql/bin/psql -U postgres -d test_knn -c "show aqo.mode;"

for ((i=1; i<=REPEAT; i++))
do
	echo "Iteration: $i"
  for FILE in "${FILES[@]}"
  do
	echo "Iter ${i}"
	/usr/local/pgsql/bin/psql -d test_knn -U postgres -f "/home/postgres/join-order-benchmark/$FILE"
  done
done
