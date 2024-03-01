
temperature=0.0
top_p=1.0
max_tokens=100
doc_top_n=5
batch_size=1
seed=20
limit_input=0

openai_config_path="./openai_config.txt"
input_data_path="./data/retrievalqa.jsonl"

# declare -a retrieval_modes=("adaptive_retrieval" "always_retrieval" "no_retrieval")
declare -a retrieval_modes=("no_retrieval")
# declare -a retrieval_modes=("always_retrieval")
# declare -a retrieval_modes=("adaptive_retrieval")


prompt_method="vanilla"
# prompt_method="TAARE"

declare -a model_names=("TinyLlama/TinyLlama-1.1B-Chat-v1.0" "microsoft/phi-2" "meta-llama/Llama-2-7b-chat-hf" "gpt-3.5-turbo-0125" "gpt-4-0125-preview")
# declare -a model_names=("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

declare -a data=("retrievalqa")
# declare -a data=("toolqa" "freshqa" "realtimeqa" "popqa" "triviaqa")


for retrieval_mode in "${retrieval_modes[@]}"
do 
    for model_name in "${model_names[@]}"
    do
        for data_source in "${data[@]}"
        do
            echo "================ Running: $data_source | Retrieval mode: $retrieval_mode | Seed: $seed ================"

            if [[ $model_name =~ "gpt-4" ]]; then
                echo "------ use less data for GPT4 ------"
                input_data_path="./data/retrievalqa_gpt4.jsonl"
            fi 

            output_prediction_path="./results/${retrieval_mode}/${model_name}/m=${prompt_method}/t=${temperature}/predict_${data_source}_seed${seed}.jsonl"
            output_score_path="./results/${retrieval_mode}/${model_name}/m=${prompt_method}/t=${temperature}/score_${data_source}_seed${seed}.json"
            
            echo $output_prediction_path
            echo $output_score_path
            echo $input_data_path

            python ./run_lm.py \
                --model_name $model_name \
                --data_source $data_source \
                --retrieval_mode $retrieval_mode \
                --openai_config_path $openai_config_path \
                --input_data_path $input_data_path \
                --output_score_path $output_score_path \
                --output_prediction_path $output_prediction_path \
                --temperature $temperature \
                --top_p $top_p \
                --max_tokens $max_tokens \
                --doc_top_n $doc_top_n \
                --batch_size $batch_size \
                --limit_input $limit_input \
                --prompt_method $prompt_method \
                --seed $seed

            echo ""
        done 
    done
done


