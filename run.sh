class_name=02691156
cuda=0
for j in {0..0}
do
	python NeuralTPS.py \
		--data_dir data/${class_name}/ \
		--out_dir $j/ \
		--class_idx $class_name \
		--train \
		--dataset other \
		--name $j \
		--CUDA $cuda

	for i in {1..8}
	do
		python NeuralTPS.py \
		--data_dir data/${class_name}/ \
		--out_dir $j/ \
		--class_idx $class_name \
		--dataset other \
		--name $j \
	    --index $i \
		--CUDA $cuda
	done
done
