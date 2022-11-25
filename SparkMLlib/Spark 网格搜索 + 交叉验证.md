## Spark 网格搜索 + 交叉验证

##### MulticlassClassificationEvaluator 源码

```scala
package org.apache.spark.ml.evaluation

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

/**
 * :: Experimental ::
 * Evaluator for multiclass classification, which expects two input columns: prediction and label.
 */
@Since("1.5.0")
@Experimental
class MulticlassClassificationEvaluator @Since("1.5.0") (@Since("1.5.0") override val uid: String)
  extends Evaluator with HasPredictionCol with HasLabelCol with DefaultParamsWritable {

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("mcEval"))

  /**
   * param for metric name in evaluation (supports `"f1"` (default), `"weightedPrecision"`,
   * `"weightedRecall"`, `"accuracy"`)
   * @group param
   */
  @Since("1.5.0")
  val metricName: Param[String] = {
    val allowedParams = ParamValidators.inArray(Array("f1", "weightedPrecision",
      "weightedRecall", "accuracy"))
    new Param(this, "metricName", "metric name in evaluation " +
      "(f1|weightedPrecision|weightedRecall|accuracy)", allowedParams)
  }

  /** @group getParam */
  @Since("1.5.0")
  def getMetricName: String = $(metricName)

  /** @group setParam */
  @Since("1.5.0")
  def setMetricName(value: String): this.type = set(metricName, value)

  /** @group setParam */
  @Since("1.5.0")
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  @Since("1.5.0")
  def setLabelCol(value: String): this.type = set(labelCol, value)

  setDefault(metricName -> "f1")

  @Since("2.0.0")
  override def evaluate(dataset: Dataset[_]): Double = {
    val schema = dataset.schema
    SchemaUtils.checkColumnType(schema, $(predictionCol), DoubleType)
    SchemaUtils.checkNumericType(schema, $(labelCol))

    val predictionAndLabels =
      dataset.select(col($(predictionCol)), col($(labelCol)).cast(DoubleType)).rdd.map {
        case Row(prediction: Double, label: Double) => (prediction, label)
      }
      
    // 传入metricName后进行模式匹配，调用MulticlassMetrics中的方法
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val metric = $(metricName) match {
      case "f1" => metrics.weightedFMeasure
      case "weightedPrecision" => metrics.weightedPrecision
      case "weightedRecall" => metrics.weightedRecall
      case "accuracy" => metrics.accuracy
    }
    metric
  }

  @Since("1.5.0")
  override def isLargerBetter: Boolean = true

  @Since("1.5.0")
  override def copy(extra: ParamMap): MulticlassClassificationEvaluator = defaultCopy(extra)
}

@Since("1.6.0")
object MulticlassClassificationEvaluator
  extends DefaultParamsReadable[MulticlassClassificationEvaluator] {

  @Since("1.6.0")
  override def load(path: String): MulticlassClassificationEvaluator = super.load(path)
}
```

##### 网格搜索

```scala
// 使用ParamGridBuilder构建一个用于搜索的参数网格
// 最大深度的参数设置3个，决策树数量参数设置2个，构建了一个 3 * 2 = 6 的参数网格以供交叉验证选择
val paramGrid = new ParamGridBuilder()
  .addGrid(RandomForestClassifier.maxDepth, Array(10, 15, 20))
  .addGrid(RandomForestClassifier.numTrees, Array(30, 40))
  .build()
```

##### 交叉验证

```scala
// 将 Pipeline 作为 Estimator，传入一个 CrossValidator 实例中，共同为所有 Pipeline 阶段选择参数
// CrossValidator 需要一个 Estimator、一组 Estimator ParamMap 和一个 Evaluator
// 此处的 Evaluator 是 MulticlassClassificationEvaluator
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new MulticlassClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(2)  // 设置折叠数量
```