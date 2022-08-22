gpu_tester --nodes 2 --parallel-tests 50 --job_comment laion --partition "gpu" --test_kind "ddp" --job_timeout 45 --exclude 'gpu-st-p4d-24xlarge-[66]'
