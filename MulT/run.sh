### Multimodal Setting

# MOSI unaligned
# python main.py --dataset mosi --data_path /data1/multimodal/processed_data --batch_size 32 --num_heads 10 --embed_dropout 0.2 --attn_dropout 0.2 --out_dropout 0.1 --num_epochs 100

## MOSEI unaligned
# python main.py --dataset mosei_senti --data_path /data1/multimodal/processed_data --batch_size 16 --num_heads 10 --embed_dropout 0.3 --attn_dropout 0.1 --out_dropout 0.1 --num_epochs 20

## MOSI aligned
# python main.py --dataset mosi --data_path /data1/multimodal/processed_data --batch_size 32 --num_heads 10 --embed_dropout 0.2 --attn_dropout 0.2 --out_dropout 0.1 --num_epochs 100 --aligned

## MOSEI aligned
# python main.py --dataset mosei_senti --data_path /data1/multimodal/processed_data --batch_size 16 --num_heads 10 --embed_dropout 0.3 --attn_dropout 0.1 --out_dropout 0.1 --num_epochs 20 --aligned


### Unimodal Setting

## modality = 'text' or 'audio' or 'visual'
## MOSI - text
# python unimodal/main.py --dataset mosi --data_path /data1/multimodal/processed_data --batch_size 32 --num_heads 10 --embed_dropout 0.2 --attn_dropout 0.2 --out_dropout 0.1 --num_epochs 100 --modality 'text'

## MOSI - visual
# python unimodal/main.py --dataset mosi --data_path /data1/multimodal/processed_data --batch_size 32 --num_heads 10 --embed_dropout 0.2 --attn_dropout 0.2 --out_dropout 0.1 --num_epochs 100 --modality 'visual'

## MOSI - audio
# python unimodal/main.py --dataset mosi --data_path /data1/multimodal/processed_data --batch_size 32 --num_heads 10 --embed_dropout 0.2 --attn_dropout 0.2 --out_dropout 0.1 --num_epochs 100 --modality 'audio'

## MOSEI - text
python unimodal/main.py --dataset mosei_senti --data_path /data1/multimodal/processed_data --batch_size 16 --num_heads 10 --embed_dropout 0.3 --attn_dropout 0.1 --out_dropout 0.1 --num_epochs 20 --modality 'text'

## MOSEI - visual
python unimodal/main.py --dataset mosei_senti --data_path /data1/multimodal/processed_data --batch_size 16 --num_heads 10 --embed_dropout 0.3 --attn_dropout 0.1 --out_dropout 0.1 --num_epochs 20 --modality 'visual'

## MOSEI - audio
python unimodal/main.py --dataset mosei_senti --data_path /data1/multimodal/processed_data --batch_size 16 --num_heads 10 --embed_dropout 0.3 --attn_dropout 0.1 --out_dropout 0.1 --num_epochs 20 --modality 'audio'
