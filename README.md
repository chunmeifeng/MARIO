# MARIO
# Deep multi-modal aggregation network for MR image reconstruction with auxiliary modality

## Dependencies
* numpy==1.18.5
* scikit_image==0.16.2
* torchvision==0.8.1
* torch==1.7.0
* runstats==1.8.0
* pytorch_lightning==1.0.6
* h5py==2.10.0
* PyYAML==5.4

Our code is based on the fastMRI, more details can be find at https://github.com/facebookresearch/fastMRI.

**Train**
```bash
cd experimental/MANet/
sbatch job.sh
```

**Change other arguments that you can train your own model.**

**The detailed parameter settings can be find in our arXiv paper.**

Citation

If you use our code in your project, please cite the arXiv papers:

```
@inproceedings{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
@article{feng2021multi,
  title={Deep multi-modal aggregation network for MR image reconstruction with auxiliary modality},
  author={Feng, Chun-Mei and Fu, Huazhu and Zhou, Tianfei and Xu, Yong and Shao, Ling and Zhang, David},
  journal={arXiv preprint arXiv:2110.08080},
  year={2021}
}
```
