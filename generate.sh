# Generates texts with FNN and FNN_tied model

# FNN
python generate.py --cuda --model FNN --checkpoint saved_model/FNN.pt \
    --seed 1 --bptt 10 --outf gen_txt/FNN_gen_1.txt
python generate.py --cuda --model FNN --checkpoint saved_model/FNN.pt \
    --seed 2 --bptt 10 --outf gen_txt/FNN_gen_2.txt
python generate.py --cuda --model FNN --checkpoint saved_model/FNN.pt \
    --seed 3 --bptt 10 --outf gen_txt/FNN_gen_3.txt

# FNN tied
python generate.py --cuda --model FNN --checkpoint saved_model/FNN_tied.pt \
    --seed 1 --bptt 10 --outf gen_txt/FNN_tied_gen_1.txt
python generate.py --cuda --model FNN --checkpoint saved_model/FNN_tied.pt \
    --seed 2 --bptt 10 --outf gen_txt/FNN_tied_gen_2.txt
python generate.py --cuda --model FNN --checkpoint saved_model/FNN_tied.pt \
    --seed 3 --bptt 10 --outf gen_txt/FNN_tied_gen_3.txt
