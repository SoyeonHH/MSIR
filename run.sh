## Experiments (model_name: TFN, Glove, Facet, COVAREP)
# python src/main.py --dataset mosi --model_name COVAREP --learning_rate 0.0005 --num_epochs 50

## TFN
# python TFN/main.py --dataset mosi --model_name TFN

## MMIM
# python MMIM/main.py --dataset mosei

## MISA
python MISA/src/train.py --data mosi

## MAG
# python MAG/multimodal_driver.py --dataset mosei

# MME2E
# python MME2E/main.py
# python MME2E/main.py --model=mme2e