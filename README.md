# Word-level language model FNN

## Training, Testing, and Generating New Text
This example trains a FNN on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
```
# Train a FNN Model
mkdir saved_model
CUDA_VISIBLE_DEVICES=6 python main.py --cuda --epochs 15 \
  --model FNN --bptt 10 --dropout 0 --save saved_model/FNN.pt

# Train a FNN Model with shared input-output layer
CUDA_VISIBLE_DEVICES=7 python main.py --cuda --epochs 15 \
  --model FNN --bptt 10 --dropout 0 --tied --save saved_model/FNN_tied.pt
```
The trained models are automatically evaluated on the validation and test set,
with Perplexity and Spearman Correlation metrics, in `main.py`.

The trained model can then be used by the generate script to generate new text.
```
bash generate.sh
```

## Program Output
