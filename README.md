# 撮って出しを再現する3DLUTの学習

## 使用方法

### 動作環境・環境構築
Python3

```
pip install -r requirements
cd trilinear_cpp
python setup.py install
```

### 学習の実行

#### データの前準備

撮って出しを再現する3D LUTの学習には以下のデータが必要になります。

* カメラで撮影したRAWを現像ソフトでJPEGとして出力したファイル
* 上のRAWに対応する撮って出しJPEGのファイル

まず、これらの2つの写真ファイルのペアを大量に準備してください。
目安として、1万枚の写真のペアを利用すれば学習が行えることを確認しています。データ数が少なくても学習の実行自体は可能ですが、撮って出し再現のクオリティに影響がある可能性があります。

これらの写真データを、以下のディレクトリ構成で配置してください。```raw```ディレクトリにはRAWから出力したJPEGファイル、```jpeg```ディレクトリには撮って出しのJPEGファイルを配置します。なお、対応するそれぞれの写真のファイル名は同じである必要があります。

```
data/
    raw/
        L1000001.jpg
        L1000002.jpg
        ...
    jpeg/
        L1000001.jpg
        L1000002.jpg
        ...
```

学習にかかる時間を削減するために、各写真は解像度をあらかじめ落として保存することをお勧めします。こちらの実験では、縦横1/32まで解像度を落として保存しました。この解像度はLUTの学習への影響は小さいことを確認しています。

#### train_lut.pyの実行

以下のコマンドを実行してください。```data/dir/path```はデータの前準備で用意したディレクトリのパス、```ouptut/dir/path```は学習結果を保存するディレクトリのパスです。

```
python train_lut.py --lut_dim 33 --dataset_path data/dir/path --output_dir output/dir/path --n_epochs 20 --batch_size 32 --n_cpu 2
``````

#### 学習結果のLUTをcube形式に変換

```
python utils/LUT2cube.py output/dir/path/LUT_20.pth output/dir/path/LUT.cube
```

#### 生成したLUTをLightroomで利用する方法

以下の記事の1.3が参考になります。

[LightroomのプロファイルをPhotoshopで簡単作成！オリジナルLUTで、自分のスタイルの風景写真に仕上げよう](https://xico.media/tutorials/make-original-profile-tips/#Photoshop)