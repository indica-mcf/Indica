#!/bin/bash

for i in {0..50}
do
	res="$(pytest tests/unit/converters/test_time.py::test_unchanged_attrs)"
	if [ $? -eq 1 ]
	then
		echo "$res" >> fail_log.txt
	fi
done
