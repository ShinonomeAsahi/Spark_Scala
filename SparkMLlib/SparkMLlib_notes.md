# SparkMLlib

## linalg - 基本数据类型

### 本地矢量 Local Vector

- 稠密向量 Dense Vector

	- newDenseVector(values: Array[Double])

		- 获取第i个元素的值

			- apply(i: Int): Double

		- 获取值最大元素的索引

			- argmax: Int

		- 转换为ml库向量

			- asML: ml.linalg.DenseVector

		- 返回使用存储空间更少的形式（稠密或稀疏）

			- compressed: Vector

		- 对向量深拷贝

			- copy: DenseVector

		- 计算与另一向量的点积（@Since( "3.0.0" )）

			- dot(v: Vector): Double

		- 比较，返回布尔值

			- equals(other: Any): Boolean

		- 将函数 f 应用于向量的所有活动元素

			- foreachActive(f: (Int, Double) ⇒ Unit): Unit

		- 返回向量哈希值

			- hashCode(): Int

		- 返回活动元素数量

			- numActives: Int

		- 返回非 0 元素数量

			- numNonzeros: Int

		- 返回向量中元素数量

			- size: Int

		- 转换为双精度型数组

			- toArray: Array[Double]

		- 转换为JSON

			- toJson: String

		- 转换为稀疏向量

			- toSparse: SparseVector

		- 转换为字符串

			- toString(): String

		- 向量值

			- values: Array[Double]

- 稀疏向量 Sparse Vector

	- Vectors.sparse([长度，非0向量索引，索引对应的向量值])

		- 获取第i个元素的值

			- apply(i: Int): Double

		- 获取值最大元素的索引

			- argmax: Int

		- 转换为ml库向量

			- asML: ml.linalg.SparseVector

		- 返回使用存储空间更少的形式（稠密或稀疏）

			- compressed: Vector

		- 对向量深拷贝

			- copy: DenseVector

		- 计算与另一向量的点积（@Since( "3.0.0" )）

			- dot(v: Vector): Double

		- 比较，返回布尔值

			- equals(other: Any): Boolean

		- 将函数 f 应用于向量的所有活动元素

			- foreachActive(f: (Int, Double) ⇒ Unit): Unit

		- 返回向量哈希值

			- hashCode(): Int

		- 返回活动元素数量

			- numActives: Int

		- 返回非 0 元素数量

			- numNonzeros: Int

		- 返回向量中元素数量

			- size: Int

		- 转换为双精度型数组

			- toArray: Array[Double]

		- 转换为稠密向量

			- toDense: DenseVector

		- 转换为JSON

			- toJson: String

		- 转换为字符串

			- toString(): String

		- 向量值

			- values: Array[Double]

		- 非 0 向量索引

			- indices: Array[Int]

	- Vectors.sparse(3,Array(0,2),Array(2.0,8.0))
	- Vectors.sparse(3,Seq((0,2.0),(2,8.0)))

### 标记点 Labeled Point

- 创建一个被标记为1.0的稠密向量标记点

	- LabeledPoint(1.0,Vectors.dense(2.0,0.0,8.0))

		- 特征

			- features: Vector

		- 标记

			- label: Double

		- 获取特征

			- getFeatures: Vector

		- 获取标记

			- getLabel: Double

		- 转换为字符串

			- getLabel: Double

- 创建一个被标记为0的稀疏向量

	- LabeledPoint(0.0,Vectors.sparse(3,Seq((0,2.0),(2,8.0))))

		- 特征

			- features: Vector

		- 标记

			- label: Double

		- 获取特征

			- getFeatures: Vector

		- 获取标记

			- getLabel: Double

		- 转换为字符串

			- getLabel: Double

### 本地矩阵 Local Martix

- 稠密矩阵 Dense Matrix

	- newDenseMatrix(numRows: Int, numCols: Int, values: Array[Double])

		- 获取第i个元素的值

			- apply(i: Int): Double

		- 转换为ml库矩阵

			- asML: ml.linalg.DenseMatrix

		- 返回列向量迭代器

			- colIter: Iterator[Vector]

		- 返回行向量迭代器

			- rowIter: Iterator[Vector]

		- 对矩阵深拷贝

			- copy: DenseMatrix

		- 比较，返回布尔值

			- equals(other: Any): Boolean

		- 返回矩阵哈希值

			- hashCode(): Int

		- 返回活动元素数量

			- numActives: Int

		- 返回非 0 元素数量

			- numNonzeros: Int

		- 返回向量中元素数量

			- size: Int

		- 转换为双精度型数组

			- toArray: Array[Double]

		- 转换为稀疏矩阵

			- toSparse: SparseMatrix

		- 转换为字符串

			- toString(): String

		- 矩阵乘法

			- multiply(y: Vector/DenseVector/DenseVector): DenseMatrix/DenseVector

		- 获取列数

			- numCols(): Int

		- 获取行数

			- numRows(): Int

		- 矩阵转置

			- transpose: DenseMatrix

		- 是否被转置

			- isTransposed: Boolean

- 稀疏矩阵 Sparse Matrix

	- Matrices.sparse(3,2,Array(0,1,3),Array(0,2,1),Array(9,6,8))

	  创建一个3行2列的稀疏矩阵[ [9.0,0.0], [0.0,8.0], [0.0,6.0]]
	  第一个数组参数表示列指针，即每一列元素的开始索引值
	  第二个数组参数表示行索引，即对应的元素是属于哪一行
	  第三个数组即是按列先序排列的所有非零元素，通过列指针和行索引即可判断每个元素所在的位置
	  
		- 获取第i个元素的值

			- apply(i: Int): Double

		- 转换为ml库矩阵

			- asML: ml.linalg.DenseMatrix

		- 返回列向量迭代器

			- colIter: Iterator[Vector]

		- 对矩阵深拷贝

			- copy: DenseMatrix

		- 比较，返回布尔值

			- equals(other: Any): Boolean

		- 返回矩阵哈希值

			- hashCode(): Int

		- 返回活动元素数量

			- numActives: Int

		- 返回非 0 元素数量

			- numNonzeros: Int

		- 返回向量中元素数量

			- size: Int

		- 转换为双精度型数组

			- toArray: Array[Double]

		- 转换为稠密矩阵

			- toDense: DenseMatrix

		- 转换为字符串

			- toString(): String

		- 矩阵乘法

			- multiply(y: Vector/DenseVector/DenseVector): DenseMatrix/DenseVector

		- 返回行向量迭代器

			- rowIter: Iterator[Vector]

		- 矩阵转置

			- transpose: DenseMatrix

		- 获取列数

			- numCols(): Int

		- 获取行数

			- numRows(): Int

		- 是否被转置

			- isTransposed: Boolean

### 分布式矩阵 Distributed Matrix

- 行矩阵 Row Matrix

	- sc.parallelize(Array(dv1,dv2)) =>  new RowMatrix(rows)

		- 使用抽样方法计算矩阵列间相似度

			- columnSimilarities(threshold: Double): CoordinateMatrix

		- 计算列间余弦相似度

			- columnSimilarities(): CoordinateMatrix

		- 按列统计数据

			- computeColumnSummaryStatistics(): MultivariateStatisticalSummary

		- 计算协方差矩阵

			- computeCovariance(): Matrix

		- 计算格拉姆矩阵

			- computeGramianMatrix(): Matrix

		- 计算前K个主成分

			- computePrincipalComponents(k: Int): Matrix

		- 计算前 k 个主成分和由每个主成分解释的方差比例向量

			- computePrincipalComponentsAndExplainedVariance(k: Int): (Matrix, Vector)

		- 奇异值分解

			- computeSVD(k: Int, computeU: Boolean = false, rCond: Double = 1e-9)

		- 矩阵相加

			- multiply(B: Matrix): RowMatrix

		- 获取列数

			- numCols(): Long

		- 获取行数

			- numRows(): Long

		- 获取行

			- rows: RDD[Vector]

		- 计算行矩阵的QR分解

			- tallSkinnyQR(computeQ: Boolean = false): QRDecomposition[RowMatrix, Matrix]

- 索引行矩阵 Indexed Row Matrix

	- IndexedRow(1,dv1) => sc.parallelize(Array(idxr1,idxr2)) => new IndexedRowMatrix(idxrows)

		- 计算列间余弦相似度

			- columnSimilarities(): CoordinateMatrix

		- 计算格拉姆矩阵

			- computeGramianMatrix(): Matrix

		- 奇异值分解

			- computeSVD(k: Int, computeU: Boolean = false, rCond: Double = 1e-9)

		- 矩阵相加

			- multiply(B: Matrix): IndexedRowMatrix

		- 获取列数

			- numCols(): Long

		- 获取行数

			- numRows(): Long

		- 获取行

			- rows: RDD[IndexedRow]

		- 转换为分块矩阵

			- toBlockMatrix(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix

		- 转换为坐标矩阵

			- toCoordinateMatrix(): CoordinateMatrix

		- 转换为行矩阵

			- toCoordinateMatrix(): CoordinateMatrix

- 坐标矩阵 Coordinate Matrix

	- new MatrixEntry(0,1,0.5) => sc.parallelize(Array(ent1,ent2)) =>  new CoordinateMatrix(entries)

		- 获取矩阵

			- valentries: RDD[MatrixEntry]

		- 获取列数

			- numCols(): Long

		- 获取行数

			- numRows(): Long

		- 转换为分块矩阵

			- toBlockMatrix(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix

		- 转换为索引行矩阵

			- toIndexedRowMatrix(): IndexedRowMatrix

		- 转换为行矩阵

			- toRowMatrix(): RowMatrix

		- 矩阵转置

			- transpose(): CoordinateMatrix

- 分块矩阵 Block Matrix

	- new MatrixEntry(3,3,1) =>  sc.parallelize => new CoordinateMatrix(entries) => coordMat.toBlockMatrix(2,2)

		- 矩阵相接

			- add(other: BlockMatrix): BlockMatrix

		- 获取块

			- valblocks: RDD[((Int, Int), Matrix)]

		- 缓存底层RDD

			- cache(): BlockMatrix.this.type

		- 获取每块的列

			- colsPerBlock: Int

		- 矩阵相加

			- multiply(other: BlockMatrix, numMidDimSplits: Int): BlockMatrix

		- 矩阵的列数

			- numColBlocks: Int

		- 矩阵的行数

			- numRows(): Long

		- 将底层RDD持久化到硬盘

			- persist(storageLevel: StorageLevel): BlockMatrix.this.type

		- 获取每块的行

			- valrowsPerBlock: Int

		- 删去给定的块

			- subtract(other: BlockMatrix): BlockMatrix

		- 转换为坐标矩阵

			- toCoordinateMatrix(): CoordinateMatrix

		- 转换为索引行矩阵

			- toIndexedRowMatrix(): IndexedRowMatrix

		- 转换为本地矩阵

			- toLocalMatrix(): Matrix

		- 矩阵转置

			- transpose: BlockMatrix

		- 验证分块矩阵信息

			- validate(): Unit

## stat - 统计信息

### 概括统计数据 Summary Statistics

- 调用colStats()方法，得到一个MultivariateStatisticalSummary类型的变量： val summary = Statistics.colStats(data)

	- 列的大小 summary.count
	- 每列的平均值 summary.mean
	- 每列的方差  summary.variance
	- 每列最大、最小值 summary.max/min
	- 每列L1、L2范数 summary.normL1/normL2
	- 每列非0向量个数 summary.numNonzeros

### 相关性 Correlations

/**
  * 相关性
  * 相关系数是用以反映变量之间相关关系密切程度的统计指标
  * 相关系数绝对值越大（值越接近1或者-1）
  * 当取值为0表示不相关
  * 取值为(0~-1]表示负相关
  * 取值为(0, 1]表示正相关
  * */

- 皮尔逊相关系数 Pearson

	- Statistics.corr(seriesX,seriesY,"pearson")

- 斯皮尔曼等级相关系数 Spearman

	- Statistics.corr(seriesX,seriesY,"spearman")

### 分层取样 Stratified Sampling

- sampleByKey

	- d1.sampleByKey(withReplacement = false, fractions, 1)

- sampleByKeyExact

	- d1.sampleByKeyExact(withReplacement = false, fractions, 1)

### 假设检验 Hypothesis Testing

- 皮尔森卡方检测 Pearson’s chi-squared tests

	- Statistics.chiSqTest(mat)

- 柯尔莫哥洛夫-斯米尔诺夫检验（KS检验）

	- Statistics.kolmogorovSmirnovTest(data, myCDF)

### 随机数据生成 Random Data Generation

- normalRDD(sc,10000000L, 10)
- normalRDD[ sc，生成数量，分区数 ]

### 核密度估计  Kernel Density Estimation

- val kd = new KernelDensity()
      .setSample(data3)
      .setBandwidth(3.0)

	- kd.estimate(Array(-1.0, 2.0, 5.0))

## feature - 特征工程

### 文本特征提取

- TF-IDF（词频-逆文本频率）

	- HashingTF是一个特征词集的转换器（Transformer），它可以将这些集合转换成固定长度的特征向量

		- val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordData)

	- IDF是一个适合数据集并生成IDFModel的评估器，IDFModel获取特征向量并缩放每列

		-   val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

- Word2Vec

	- 将每个单词映射到一个唯一的固定大小的向量

		- val word2vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2vec.fit(documentDF)
    val result = model.transform(documentDF)

- CountVectorizer

	- 将文本文档集合转换为计数向量

		-  val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)

### 特征变换

- Tokenizer（分词器）

	- 将文本划分为独立个体（通常是单词）

		- val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val regexTokenizer = new RegexTokenizer()
  .setInputCol("sentence")
  .setOutputCol("words")
  .setPattern("\\W")

- StopWordsRemover

	- 去除停用词

		- val remover = new StopWordsRemover()
  .setInputCol("raw")
  .setOutputCol("filtered")

- n-gram

	- 将输入转换为n-gram

		- 
val ngram = new NGram().setInputCol("words").setOutputCol("ngrams")

- Binarizer （二值化）

	- 根据阀值将连续数值特征转换为0-1特征

		- val binarizer: Binarizer = new Binarizer()
  .setInputCol("feature")
  .setOutputCol("binarized_feature")
  .setThreshold(0.5)

- PCA （主成分分析）

	- 使用正交转换从一系列可能相关的变量中提取线性无关变量集，提取出的变量集中的元素称为主成分

		- val pca = new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(3)
  .fit(df)

- PolynomialExpansion （多项式扩展）

	- 通过产生n维组合将原始特征将特征扩展到多项式空间

		- val polynomialExpansion = new PolynomialExpansion()
  .setInputCol("features")
  .setOutputCol("polyFeatures")
  .setDegree(3)

- DCT （离散余弦变换）

	- 类似于离散傅立叶变换，但只使用实数

		- val dct = new DCT()
  .setInputCol("features")
  .setOutputCol("featuresDCT")
  .setInverse(false)

- 归一化

	- Normalizer

		- 作用范围是每一行，使每一个行向量的范数变换为一个单位范数

			- val normalizer = new Normalizer()
.setInputCol("features")
.setOutputCol("normFeatures")
.setP(1.0)

				- val l1NormData = normalizer.transform(dataFrame) // 一阶
				- val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity) // 无穷阶

	- StandardScaler

		- 处理的对象是每一列，也就是每一维特征，将特征标准化为单位标准差或是0均值，或是0均值单位标准差

			- val scaler = new StandardScaler()
.setInputCol("features")
.setOutputCol("scaledFeatures")
.setWithStd(true)
.setWithMean(false)

				- val scalerModel = scaler.fit(dataFrame)

					- val scaledData = scalerModel.transform(dataFrame)

	- MinMaxScaler

		- 作用同样是每一列，即每一维特征。将每一维特征线性地映射到指定的区间，通常是[0, 1]

			- val scaler = new MinMaxScaler()
.setInputCol("features")
.setOutputCol("scaledFeatures")

				- val scalerModel = scaler.fit(dataFrame)

					- val scaledData = scalerModel.transform(dataFrame)

	- MaxAbsScaler

		- 将每一维的特征变换到[-1, 1]闭区间上，通过除以每一维特征上的最大的绝对值，它不会平移整个分布，也不会破坏原来每一个特征向量的稀疏性。

			- val scaler = new MaxAbsScaler()
.setInputCol("features")
.setOutputCol("scaledFeatures")

				- val scalerModel = scaler.fit(dataFrame)

					- val scaledData = scalerModel.transform(dataFrame)

- StringIndexer

	- 将字符串标签编码为标签指标

		- val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

- IndexToString

	- 将指标标签映射回原始字符串标签

		- val converter = new IndexToString()
  .setInputCol("categoryIndex")
  .setOutputCol("originalCategory")

- OneHotEncoder - 独热编码

	- 将标签指标映射为二值向量

		- val encoder = new OneHotEncoder()
  .setInputCol("categoryIndex")
  .setOutputCol("categoryVec")

- VectorIndexer

	- 自动识别哪些特征是类别型的，并且将原始值转换为类别指标

		- val indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexed")
  .setMaxCategories(10)

- Bucketizer

	- 将一列连续的特征转换为特征区间

		- val bucketizer = new Bucketizer()
  .setInputCol("features")
  .setOutputCol("bucketedFeatures")
  .setSplits(splits)

- ElementwiseProduct

	- 按提供的“weight”向量，返回与输入向量元素级别的乘积

		- val transformer = new ElementwiseProduct()
  .setScalingVec(transformingVector)
  .setInputCol("vector")
  .setOutputCol("transformedVector")

- SQLTransformer

	- 用来转换由SQL定义的陈述

		- val sqlTrans = new SQLTransformer().setStatement(
  "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")

- VectorAssembler - 向量装配器

	- 将给定的若干列合并为一列向量

		- val assembler = new VectorAssembler()
  .setInputCols(Array("hour", "mobile", "userFeatures"))
  .setOutputCol("features")

- QuantileDiscretizer

	- 讲连续型特征转换为分级类别特征

		- val discretizer = new QuantileDiscretizer()
  .setInputCol("hour")
  .setOutputCol("result")
  .setNumBuckets(3)

### 特征选择

- VectorSlicer （向量机）

	- 输出一个新的特征向量与原始特征的子阵列

		-   val slicer = new VectorSlicer()
      .setInputCol("userFeatures")
      .setOutputCol("features")
    slicer.setIndices(Array(1)).setNames(Array("f3"))

- RFormula （R公式）

	- 选择由R模型公式指定的列

		-   val formula = new RFormula()
      .setFormula("clicked ~ country + hour")
      .setFeaturesCol("features")
      .setLabelCol("label")

- ChiSqSelector（卡方特征选择）

	- 使用卡方独立测试来决定选择哪些特征

		-   val selector = new ChiSqSelector()
      .setNumTopFeatures(1)
      .setFeaturesCol("features")
      .setLabelCol("clicked")
      .setOutputCol("selectedFeatures")

## 分类和回归

### 分类

- LogisticRegression（逻辑回归）
- DecisionTreeClassifier （决策树）
- Random Forest（随机森林）

	- numTrees (default = 20)

		- 越大越好

	- maxDepth (default = 5)

		- 按5步长递增

- GBTClassifier（梯度迭代树）
- MultilayerPerceptronClassifier（多层感知机）
- One-vs-Rest（一对多分类器）
- MulticlassClassificationEvaluator（朴素贝叶斯）

### 回归

- GLMs（广义线性模型）
- DecisionTreeRegressor（决策树回归）
- RandomForestRegressor（随机森林回归）
- GBDT（梯度迭代树回归）
- AFTSurvivalRegression（生存回归）
- IsotonicRegression（保序回归算法）

## tree - 树

### DecisionTree - 决策树

### GradientBoostedTrees - 梯度提升树

### RandomForest - 随机森林

## 聚类

### K-means（K均值算法）

### LDA（文档主题生成模型）

### BisectingKMeans（二分K均值）

### GMM（混合高斯模型）

## 协同过滤

### UserCF

### ItemCF

### 相似度计算

- 欧几里得距离

	- 以目标绝对距离作为衡量

- 皮尔逊相关系数
- 余弦相似度

	- 以目标差异的大小作为衡量

- Jaccard相似系数

### ALS

## 模型选择和调试

### CrossValidator（交叉验证）

### TrainValidationSplit（训练检验分裂）

## 降维

### 奇异值分解 SVD

- 创建样本矩阵 Vectors.sparse/dense

	- 将样本矩阵生成RDD sc.parallelize

		- 创建行矩阵 new RowMatrix()

			- 计算前5个奇异值和对应的奇异向量
mat.computeSVD(5,computeU = true)

				- 左奇异向量
svd.U
				- 右奇异向量
svd.V
				- 奇异值
 svd.s

### 主成分分析 PCA

- 计算前4个主成分
 mat.computePrincipalComponents(4)

## spark.ml

### Pipeline - 流程化

- Transformers - 转换器

	- Transformers.transform

- Estimators - 预测器

	- Estimators.fit

- val pipeline = new Pipeline()
  .setStages(Array(tokenizer, hashingTF, lr))

	- val model = pipeline.fit(training)

		- model.transform(test)

## breeze.linalg

### import breeze.linalg._ / breeze.numerics._

- DenseVector

	- 创建全零向量

		- DenseVector.zeros[Double](n)

	- 创建全一向量

		- DenseVector.ones[Double](n)

	- 按数值填充向量

		- DenseVector.fill(n){5.0}

	- 创建线性等分向量

		- DenseVector.linspace(start, stop, numvals)

	- 按照行创建向量

		- DenseVector(1, 2, 3, 4)

- DenseMatrix

	- 创建全零矩阵

		- DenseMatrix.zeros[Double](n,m)

	- 创建单位矩阵

		- DenseMatrix.eye[Double](n)

	- 创建对角矩阵

		- diag(DenseVector(1.0, 2.0, 3.0))

	- 按照行创建矩阵

		- DenseMatrix((1.0, 2.0), (3.0, 4.0))

