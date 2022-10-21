rm -rf /p/home/jusers/beaumont1/juwels/gpu-tester/results

EXCLUDE=`python /p/home/jusers/beaumont1/juwels/gpu-tester/a.py`
echo $EXCLUDE
gpu_tester --nodes 4 --parallel-tests 20 --job_comment laion --partition "booster" --test_kind "ddp" --job_timeout 180 --gpu_per_node 4 --exclude $EXCLUDE
