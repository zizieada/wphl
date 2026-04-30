# Evaluation Results

This folder contains CSV files with **subjective quality scores** (e.g. Mean Opinion Scores) and **objective metric outputs** for the datasets used in the paper.

> **We are not the authors of these datasets.** The scores provided here are reproduced or derived solely to support reproducibility of the results presented in the paper. If you use any of these scores in your own work, please cite the original dataset papers listed below.

---

## Datasets and Citations

Please cite the following works when using the corresponding data:

**KADID-10k**
```bibtex
@InProceedings{Lin_2019,
  author    = {Hanhe Lin and Vlad Hosu and Dietmar Saupe},
  booktitle = {2019 Eleventh International Conference on Quality of Multimedia Experience ({QoMEX})},
  title     = {{KADID-10k: A Large-scale Artificially Distorted IQA Database}},
  year      = {2019},
  month     = jun,
  publisher = {{IEEE}},
  doi       = {10.1109/qomex.2019.8743252},
}
```

**CSIQ**
```bibtex
@Article{Larson_2010,
  author    = {Eric Cooper Larson and Damon Michael Chandler},
  journal   = {Journal of Electronic Imaging},
  title     = {{Most apparent distortion: full-reference image quality assessment and the role of strategy}},
  year      = {2010},
  pages     = {1--21},
  volume    = {19},
  doi       = {10.1117/1.3267105},
  publisher = {SPIE},
}
```

**TID2013**
```bibtex
@Article{Ponomarenko_2015,
  author    = {Ponomarenko, Nikolay and Jin, Lina and Ieremeiev, Oleg and Lukin, Vladimir and Egiazarian, Karen and Astola, Jaakko and Vozel, Benoit and Chehdi, Kacem and Carli, Marco and Battisti, Federica and Jay Kuo, C.-C.},
  journal   = {Signal Processing: Image Communication},
  title     = {{Image database TID2013: Peculiarities, results and perspectives}},
  year      = {2015},
  issn      = {0923-5965},
  month     = jan,
  pages     = {57--77},
  volume    = {30},
  doi       = {10.1016/j.image.2014.10.009},
  publisher = {Elsevier BV},
}
```

**LIVE**
```bibtex
@Article{Sheikh_2006a,
  author    = {Sheikh, H.R. and Sabir, M.F. and Bovik, A.C.},
  journal   = {IEEE Transactions on Image Processing},
  title     = {A Statistical Evaluation of Recent Full Reference Image Quality Assessment Algorithms},
  year      = {2006},
  issn      = {1057-7149},
  month     = nov,
  number    = {11},
  pages     = {3440--3451},
  volume    = {15},
  doi       = {10.1109/tip.2006.881959},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
}
```

**MDID**
```bibtex
@Article{Sun_2017,
  author    = {Sun, Wen and Zhou, Fei and Liao, Qingmin},
  journal   = {Pattern Recognition},
  title     = {MDID: A multiply distorted image database for image quality assessment},
  year      = {2017},
  issn      = {0031-3203},
  month     = jan,
  pages     = {153--168},
  volume    = {61},
  doi       = {10.1016/j.patcog.2016.07.033},
  publisher = {Elsevier BV},
}
```

**VCL@FER**
```bibtex
@Article{Zaric_2012,
  author    = {Zarić, Anđela and Tatalović, Nenad and Brajković, Nikolina and Hlevnjak, Hrvoje and Lončarić, Matej and Dumić, Emil and Grgić, Sonja},
  journal   = {Automatika},
  title     = {VCL@FER Image Quality Assessment Database},
  year      = {2012},
  issn      = {1848-3380},
  month     = jan,
  number    = {4},
  pages     = {344--354},
  volume    = {53},
  doi       = {10.7305/automatika.53-4.241},
  publisher = {Informa UK Limited},
}
```

**QADS**
```bibtex
@Misc{Zhou_2024,
  author    = {Zhou, Fei Zhou and Yao, Rongguo Yao and Liu, Bozhi Liu and Qiu, Guoping Qiu},
  title     = {Visual Quality Assessment for Super-Resolved Images: Database and Method},
  year      = {2024},
  copyright = {Creative Commons Attribution 4.0 International},
  doi       = {10.21227/KN1D-1984},
  publisher = {IEEE DataPort},
}
```

**JPEG AI**
```bibtex
@InProceedings{Upenik_2021,
  author    = {Upenik, Evgeniy and Testolina, Michela and Ascenso, Joao and Pereira, Fernando and Ebrahimi, Touradj},
  booktitle = {2021 International Conference on Visual Communications and Image Processing (VCIP)},
  title     = {Large-Scale Crowdsourcing Subjective Quality Evaluation of Learning-Based Image Coding},
  year      = {2021},
  month     = dec,
  pages     = {1--5},
  publisher = {IEEE},
  doi       = {10.1109/vcip53242.2021.9675314},
}
```

---

## Objective Metrics

The objective metric scores were computed using publicly available implementations (mainly those available in [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)). If you use these metric values directly, please also check and cite the original metric papers where appropriate.