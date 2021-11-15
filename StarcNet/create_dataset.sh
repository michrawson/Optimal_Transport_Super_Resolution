# Script for creating npy files with candidate slices and labels
# [GPS - 01/26/2019]

SIZE=${2:-32}

python3 src/create_object_slices.py \
		   --slice-size $SIZE \

python3 src/create_db.py \
                   --slice-size $SIZE \
