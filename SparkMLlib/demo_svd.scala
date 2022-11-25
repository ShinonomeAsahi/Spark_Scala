import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object demo_svd {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val sparkSession: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("read mysql write hive")
      .getOrCreate()

    val demoDF = sparkSession.read.textFile("src/main/resources/a.mat")

    //
    val data: RDD[linalg.Vector] = demoDF.rdd.map(_.split(" ").map(_.toDouble)).map(line => Vectors.dense(line))

    import sparkSession.implicits._
    // 通过data创建行矩阵 rowMatrix
    val rowMatrix: RowMatrix = new RowMatrix(data)



    // 保留95%以上特征信息
    val a_svd: SingularValueDecomposition[RowMatrix, Matrix] = rowMatrix.computeSVD(8,computeU = false)

    var sum2 = 0.0
    val s = a_svd.s.toArray
    for ( i <- 0 until(s.length)) {
      val s2 = s(i)*s(i)
      sum2 += s2
    }
    var sum = 0.0
    for ( i <- 0 until(s.length)) {
      val s2 = s(i)*s(i)
      sum += s2

      println(s"******Top K 取值为:${i+1}时, svd中s的平方和的比例******")
      println(sum/sum2*100+ "%")
    }
//    val a_svd: SingularValueDecomposition[RowMatrix, Matrix] = rowMatrix.computeSVD(3)


//    println("********S********")
//    println(a_svd.s)
//    println("********V********")
//    println(a_svd.V)
//    println(a_svd.V.numRows)
//    println(a_svd.V.numCols)
//    println(a_svd.V.asML.transpose)
//    println("*************************")
//    println(a_svd.V.asML)
//    println("********U********")
//    println(a_svd.U)
//    println(a_svd.U.numRows())
//    println(a_svd.U.numCols())


  }
}
