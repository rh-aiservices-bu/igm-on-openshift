python3 server.py \
    --single_file_model=../models/sd_xl_base_1.0.safetensors \
    --use_refiner=True \
    --refiner_single_file_model=../models/sd_xl_refiner_1.0.safetensors \
    --device=enable_sequential_cpu_offload
