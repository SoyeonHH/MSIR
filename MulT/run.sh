## MOSI unaligned
python main.py --model MULT --dataset mosi --data_path /data1/multimodal/MulT/data --batch_size 128 --num_heads 10 --embed_dropout 0.2 --attn_dropout 0.2 --out_dropout 0.1 --num_epochs 100

## MOSI aligned
# python main.py --dataset mosi --data_path /data1/multimodal/MulT/data --vonly --aonly --lonly --batch_size 32 --num_heads 10 --embed_dropout 0.2 --attn_dropout 0.2 --out_dropout 0.1 --num_epochs 100 --aligned

## MOSEI setting
# python main.py --dataset mosei_senti --data_path /data1/multimodal/MulT/data --batch_size 16 --num_heads 10 --embed_dropout 0.3 --attn_dropout 0.1 --out_dropout 0.1 --num_epochs 20