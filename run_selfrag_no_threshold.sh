
retrieval_mode="adaptive_retrieval"
temperature=0.0
top_p=1.0
max_tokens=100
doc_top_n=5
seed=20
limit_input=0
model_name="selfrag/selfrag_llama2_7b"
metric="match"
input_data_path="./data/retrievalqa.jsonl"


declare -a data=("retrievalqa")

for data_source in "${data[@]}"
do
    echo "================ Running: $data_source | Retrieval mode: $retrieval_mode | Seed: $seed ================"

    output_prediction_path="./results/${retrieval_mode}/${model_name}/threshold_none/predict_${data_source}_seed$seed.jsonl"
    output_score_path="./results/${retrieval_mode}/${model_name}/threshold_none/score_${data_source}_seed$seed.json"

    echo $output_prediction_path
    echo $output_score_path

    python ./run_selfrag_no_threshold.py \
        --model_name $model_name \
        --data_source $data_source \
        --retrieval_mode $retrieval_mode \
        --input_data_path $input_data_path \
        --output_score_path $output_score_path \
        --output_prediction_path $output_prediction_path \
        --temperature $temperature \
        --top_p $top_p \
        --max_tokens $max_tokens \
        --doc_top_n $doc_top_n \
        --limit_input $limit_input \
        --metric $metric \
        --seed $seed

    echo ""
done 
