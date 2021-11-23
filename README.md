# MANet
# Multi-modal Aggregation Network for Fast MR Imaging

## Dependencies
* numpy==1.18.5
* scikit_image==0.16.2
* torchvision==0.8.1
* torch==1.7.0
* runstats==1.8.0
* pytorch_lightning==1.0.6
* h5py==2.10.0
* PyYAML==5.4

**Train**
```bash
cd experimental/SANet/
sbatch job.sh
```

Change other arguments that you can train your own model.

Citation

If you find MANet useful for your research, please consider citing the following papers:

```
@article{feng2021multi,
  title={Multi-modal Aggregation Network for Fast MR Imaging},
  author={Feng, Chun-Mei and Fu, Huazhu and Zhou, Tianfei and Xu, Yong and Shao, Ling and Zhang, David},
  journal={arXiv preprint arXiv:2110.08080},
  year={2021}
}
```
