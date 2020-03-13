# ADMM Lasso

ADMM(交互方向乗数法)でLasso回帰を実装したものです。

## How to Build

1. [Eigen](https://github.com/eigenteam/eigen-git-mirror)をダウンロードして、適切な場所に配置する。
2. `main.cpp` の `dataset_path` と `ans_path` に説明変数と目的変数をスペース区切りにしたものへのフルパスを与え、データに合わせて `dim` と `data_size` を編集する。
3. 以下のコマンドでMakefileを生成する。

```bash
cmake . -DEIGEN_DIR=/path/to/eigen
```
4. 以下のコマンドでソフトウェアをビルドする。

```bash
make
```

5. 以下のコマンドで実行してみて同じような出力が得られると正常に実行できています。 

```bash
./admm_lasso
```

```
params:　0　0 0 0.285028 -0.301916　3.07761　0 0 0　0　-1.14269　0 0
```

