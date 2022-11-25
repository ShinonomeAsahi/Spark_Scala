import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer


object demo_similarity {

  // 余弦相似度
  def cosineSimilarity(x:Array[Double], y:Array[Double]):Double = {
    //计算分子（向量的点积）
    val member = x.zip(y).map(d => d._1 * d._2).sum
    //分别计算两个向量的模长
    val temp1 = math.sqrt(x.map((math.pow(_,2))).sum)
    val temp2 = math.sqrt(y.map((math.pow(_,2))).sum)

    // 计算分母（模长的乘积）
    val dominator = temp1 * temp2

    // 计算余弦相似度
    if (dominator == 0) Double.NaN else member / (dominator * 1.0)
  }

  // 欧几里得距离
  def euclidean(x:Array[Double], y:Array[Double]):Double = {
    math.sqrt(x.zip(y).map(p => p._1 - p._2).map(d => d * d).sum)
  }

  // 皮尔逊相关系数
  def pearsonSim(x:Array[Double], y:Array[Double]):Double = {

    // 分别求两个向量之和
    val sum_vec1 = x.sum
    val sum_vec2 = y.sum

    // 分别求两个向量平方之和
    val square_sum_vec1 = x.map(x => x * x).sum
    val square_sum_vec2 = y.map(y => y * y).sum

    // 将两个向量zip
    val zipVec = x.zip(y)
    // 求两个向量点积
    val product = zipVec.map(x => x._1 * x._2).sum

    // 求分子（协方差）
    val numerator = product - (sum_vec1 * sum_vec2 / x.length)

    // 求分母（标准差）
    val dominator = math.pow((square_sum_vec1 - math.pow(sum_vec1,2) / x.length) * (square_sum_vec2 - math.pow(sum_vec2,2) / y.length),0.5)

    if (dominator == 0) Double.NaN else numerator / (dominator * 1.0)
  }

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local[*]").setAppName("Cartesian")
    val sc = new SparkContext(conf)

    println("====================")
    println("生成二维数组")
    println("====================")

    val arr1 = Array.ofDim[Double](10, 10)

    for (i <- 0 until (arr1.length)) {
      for (j <- 0 until (arr1.length) if (i != j) && (i % 2 == 0)) {
        arr1(i)(j) = 1
      }
    }

    for (i <- 0 until (arr1.length)) {
      for (j <- 0 until (arr1.length)) {
        print(arr1(i)(j) + " ")
      }
      println()
    }

    println()
    println("========== 计算余弦相似度 ==========")
    println()

    arr1.foreach{
      x => println(cosineSimilarity(arr1(0), x))
    }

    println()
    println("========== 计算欧几里得距离 ==========")
    println()

    arr1.foreach{
      x => println(euclidean(arr1(0), x))
    }

    println()
    println("========== 计算皮尔逊相关系数 ==========")
    println()

    arr1.foreach{
      x => println(pearsonSim(arr1(0), x))
    }

  }
}

