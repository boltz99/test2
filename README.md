# Insight Flow
壁からの距離や流入口からの距離などSDF(Signed Distance Function)を入力に
入力したジオメトリの準定常状態の流速分布を出力するGANs

## フォルダ構成
- lbm: 格子ボルツマン法(LBM)に基づくシミュレーションを行うAPI
- ml: LBMで生成したデータを教師データとしてGANsの訓練を行う

```
├─lbm
│  ├─main
│  ├─param
│  ├─particle
│  ├─utils
│  │  ├─geoms
│  │  ├─.ipynb_checkpoints
│  │  └─__pycache__
│  ├─cuda
│  ├─geom5
│  ├─geom
│  ├─vtk
│  │  └─2023-04-01-08-32_water-0.1_00000
│  └─calc_condition
│      └─2023-04-01-08-32_water-0.1_00000
└─ml
    └─models
        └─__pycache__
```
## 訓練方法


### 事前準備
必要なライブラリ一覧をpip install -r requirements.txtでインストールしておく。

①計算元のジオメトリパターンを生成するプログラムを実行
lbm\utils\geoms\GeomGenerator2.py

このプログラムは障害物無しのストレート流路1パターンに加え、
円柱 or 四角柱、台形を重ならないように配置を少しずつ変えたり、円柱の大きさを変えたジオメトリパターンを生成する。
1(障害物無し) + 330(種々のパターン) = 331パターンを生成する

②上記で生成したジオメトリパターンを境界条件を少しずつ変えながらシミュレータを実行
\lbm\main\generate_dataset2.py
*注意: LBMのプログラムの実行にはCUDAが必要
PyCUDAとCUDA Toolkitのインストールをしておく必要がある

目安として、シミュレーションは1条件あたりRTX 3090で約20分強かかるので、
500条件で166hour程度要する

結果はresultフォルダに格納される。

**注意：ジオメトリパターン x 境界条件の全組み合わせを実行するため、
331ジオメトリパターン x 13境界条件 = 4303のシミュレーション条件になるため、
ジオメトリを減らすか境界条件を減らして現実的な値にした方がよい。

(参考)
必須ではないがもし３次元のSIM結果を可視化したいという場合には、
\lbm\utils\analyze_res2.py
を実行することでvtkファイルを作成できる。これをParaViewで読み込むことで可視化可能

③上記でシミュレーションした結果を深層学習の訓練データyを作成
具体的には3次元のSIMデータから高さ中央付近の平面の２次元流れ場を抽出するプログラムを実行する
\lbm\utils\extract_2d_data_v3.py

*こちらは対象ファイルをglob.globで取ってきているため、
下記など必要に応じて修正が必要。
  y_file_list = glob.glob('../result/2023-03-*/result*')

実行結果はmlフォルダにdataというディレクトリが作成され、train4などの指定した名前のディレクトリが配下に作成される
ml\data\train4
この下に
ml\data\train4\y
のフォルダが作成され、これがGround Truthの流速分布のデータとなる


④シミュレーションで使用したジオメトリをSDFに変換して訓練データXを作成するプログラムを実行する
\lbm\utils\get_sdf.py

*こちらは対象ファイルをglob.globで取ってきているため、
必要に応じて修正が必要。

ml\data\train4\X
のフォルダが作成され、これが入力データXとなる


⑤GANsの訓練を行うためのプログラムを実行
ml\master_gan.py

parserで基本的にはハイパーパラメータを設定している。

parser.add_argument("--init_model", type=str, default='load') : モデルを初期化するか訓練済みを読み込むか。
init or load
loadにした場合
parser.add_argument("--param_file_path_g")
parser.add_argument("--param_file_path_d")
に設定したpthを読みに行く。

parser.add_argument("--ns_loss_type", type=float, default=2)
NS LossのType IかType IIを選択。Type IIの方が訓練が安定するのでデフォルトでは2

need_ns_loss = TrueにするとNS Lossを含めたLossの計算を行う
