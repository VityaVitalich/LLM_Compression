python /home/LLM_compression/QUIK_ternary/experiments/fake_quant/llama.py \
    --model /home/exp_results/admm_pruning/unstructured/llama2_7b_admm_0@50 \
    --path_to_act_scales /home/LLM_compression/QUIK/experiments/act_scales/Llama-2-7b-hf.pt \
    --path_to_save_quant_model /home/exp_results/quik/llama7b_3bit_quik_scales_after_admm_sparsity_0@50 \
    --fp_features 128 --a_bits 16 --w_bits 3 --w_clip --retain_sparsity --dataset wikitext2