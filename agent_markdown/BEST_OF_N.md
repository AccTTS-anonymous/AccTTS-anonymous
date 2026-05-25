Create one file in current folder.

If you run any command or change code, record your action and status (success/fail) in the log file.

If you find problems, you can modify if necessary. 

But do remember to record your modification in the log file.

Whenever you modify my code, please git.

Finally, if you find that some test cases always fail, you can skip them and record the information.



## Step 1. Collect trace results 

Dataset is MATH500.
Method is best_of_n.

Run this:
`bash scripts/run_math500_beam_sweep.sh`
This bash command should collect generation token length for #beams=128 on MATH500.

After this, you will see some files named "beams=128" in 
`/path/to/AccTTS/data/Qwen/QWen2.5-1.5B-Instruct/MATH-500`




## Step 2. Generate more trace results for other beams settings

Same dataset and method.

Run this:
`python scripts/extract_smaller_beams.py recipes/best-of-n.yaml --source-n 128 --target-ns 16 32 64`

The reason we don't need --target-ns 4, 8... is that they exist in 
`/path/to/AccTTS/data/Qwen/QWen2.5-1.5B-Instruct/MATH-500`




## Step 3. Experiment with two methods under various #beams settings

First method is FD.
Set "gemm_opt" to false and "chunk_size" to heuristic in `/path/to/AccTTS/recipes/best-of-n.yaml`
Change "n" `/path/to/AccTTS/recipes/best-of-n.yaml` to control #beams. Please test [2, 4, 8, 16, 32, 64, 128].

After this, you will see corresponding results in 
`/path/to/AccTTS/data/Qwen/QWen2.5-1.5B-Instruct/MATH-500`

Similar for testing our method.
Set "gemm_opt" to true and "chunk_size" to dynamic in `/path/to/AccTTS/recipes/best-of-n.yaml`
Change "n" `/path/to/AccTTS/recipes/best-of-n.yaml` to control #beams. Please test [2, 4, 8, 16, 32, 64, 128].

After this, you will see corresponding results in 
`/path/to/AccTTS/data/Qwen/QWen2.5-1.5B-Instruct/MATH-500`

Similar for testing vllm.
Set "gemm_opt" to false and "chunk_size" to none in `/path/to/AccTTS/recipes/best-of-n.yaml`
Change "n" `/path/to/AccTTS/recipes/best-of-n.yaml` to control #beams. Please test [16, 32, 64] since [2, 4, 8, 128] exists in `/path/to/AccTTS/data/Qwen/QWen2.5-1.5B-Instruct/MATH-500`. You can double check.