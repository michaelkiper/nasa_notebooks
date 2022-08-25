
t=$1
#step_a="python /home/makiper/Notebooks/bootstrap.py ${t} 2 -output_path /home/makiper/Bootstrap"
step_a="python /home/makiper/Notebooks/bootstrap_2.py /home/makiper/Notebooks/true_${t}_rfls.pickle -out_base /home/makiper/Bootstrap -type ${t}"

sbatch -N 1 -c 40 --mem=180G --wrap="${step_a}"

