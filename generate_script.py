import os

datasets = [
    # "PACS",
    "OfficeHome",
    # "DomainNet",
]

env_num = {
    "PACS": 4,
    "OfficeHome": 4,
    "DomainNet": 6,
}

algos = ['VIT', 'GMOE']


gpu_id = 4
for algo in algos:
    for dataset in datasets:      
        for env_iter in range(env_num[dataset]):
            log_dir = f'log/{dataset}/{algo}/domain{env_iter}'
            os.makedirs(log_dir, exist_ok=True)
            # cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python3 -m domainbed.scripts.train --data_dir=./domainbed/data/ --algorithm {algo} --dataset {dataset} --test_env {env_iter} --output_dir {log_dir} >{log_dir}/nohup.log 2>&1 &"
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python3 -m domainbed.scripts.eval --data_dir=./domainbed/data/ --algorithm {algo} --dataset {dataset} --test_env {env_iter} --output_dir {log_dir}"
            gpu_id += 1
            print(cmd)
            if gpu_id >= 8:
                gpu_id = 4