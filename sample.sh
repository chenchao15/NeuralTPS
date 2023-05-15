class_name=04401088
python sample_query_point.py \
       --input_dir /data/cc/data/ShapeNet/${class_name}/ \
       --out_dir data/${class_name}/ \
       --class_idx ${class_name} \
       --dataset shapenet \
       --CUDA 0
