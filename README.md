# XTab: Cross-Table Pretrained Transformers

XTab provides a framework for pretraining tabular transformers. Our implementation is based on [OpenML-AutoMLBenchmark](https://github.com/openml/automlbenchmark) and [AutoGluon](https://github.com/autogluon/autogluon).

The pretraining process of XTab uses a _**distributed training**_ framework provided by [OpenML-AutoMLBenchmark](https://github.com/openml/automlbenchmark). Currently, we only support pretraining using AWS EC2 instances with at least one GPU (e.g., G4dn). The following requirements have to be fullfilled in order to repeat our expermients:

* One need have access to an AWS S3 bucket (read, write and delete files). We use S3 to save the pretrained checkpoints and synchronize across EC2 instances.
* The master machine (where we run the scripts) must have the permission to access s3 and launch EC2 instances.
* One need specify the github repository (and branch) of AutoMLBenchmark and AutoGluon. We provide our code as .zip files. Users need to unzip them, upload to a public github repository (such that the EC2 instances can access and download the code), and specify the url in the AutoMLBenchmark.

We blocked our information (GitHub repo, S3 bucket, etc) for review purpose. Above steps must be taken to make the code functional. We strongly encourage users to refer to the AWS mode of [OpenML-AutoMLBenchmark](https://github.com/openml/automlbenchmark) for any questions.

Our code is implemented in `AutoGluon.multimodel`. The AutoGluon package must also be publich on GitHub for EC2 instances to download and run. XTab is developed based on AutoGluon 0.5.3. Please refer to [AutoGluon](https://github.com/autogluon/autogluon) for questions on AutoGluon installation or usage.

After configuration, the following code will run cross-table pretraining and save the pretrained backbones on S3.
```bash
rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml
python runbenchmark.py XTab_pretrain ag_pretrain mytest1h -m aws -p 520
```

## Examples:
In the paper, all finetuning experiments are also performed using a distribued setting. Here, we provide a minimal example that can be run locally. We provide the pretrained backbones after 0, 1000, and 2000 iterations in `./pretrained checkpoints`. To show the downstream performance on the Adult Income dataset. Simply do the following:
```bash
python runexample.py --pretrained_ckpts path-to-pretrained-checkpoints
```

Use `--batch_size` to specify batch size (default 128) and `--max_epochs` to specify maximum finetuning epochs (default 3). If you specify a checkpoint that does not exist, the model will train from randomly initialized weights. Results are:

pretraining steps | Test AUC | Validation AUC | Train time | Test time  
----  | ----  | ----  | ----  | ---- 
iter_0 (no pretraining) |  0.918905 | 0.922689 | 33.828741 | 2.028967
iter_1k (pretraining for 1k iterations) |  0.919736 | 0.923456 | 33.760371  | 2.052341 

