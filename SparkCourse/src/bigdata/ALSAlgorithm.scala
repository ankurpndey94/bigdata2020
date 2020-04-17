package bigdata

import org.apache.log4j._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import org.apache.spark.ml.recommendation._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

object ALSAlgorithm {
  
  // For mapping movie id with movie names
  
  def loadMovieNames() : Map[Int, String] = {
    
    // To tackle character encoding issues
    
    implicit val code = Codec("UTF-8")
    code.onMalformedInput(CodingErrorAction.REPLACE)
    code.onUnmappableCharacter(CodingErrorAction.REPLACE)

    var movieNames:Map[Int, String] = Map()
    
     val lines = Source.fromFile("movies.csv").getLines()
     for (line <- lines) {
       var fields = line.split(',')
       if (fields.length > 1) {
        movieNames += (fields(0).toInt -> fields(1))
       }
     }
    
     return movieNames
  }
  
  // Row format to feed into ALS
  case class Rating(userId: Int, movieId: Int, rating: Float)
    

  def main(args: Array[String]) {
    
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    // Make a spark session to run it via spark submit on cmd
    val spark = SparkSession
      .builder
      .appName("ALSAlgorithm")
      .master("local[*]")
      .getOrCreate()
      
    import spark.implicits._
    
    println("Loading movie names...")
    val nameDict = loadMovieNames()
 
    val data = spark.read.textFile("ratings.csv")
    
    val ratings = data.map( x => x.split(',') ).map( x => Rating(x(0).toInt, x(1).toInt, x(2).toFloat) ).toDF()
    
    // Build the recommendation model using Alternating Least Squares
    println("\nTraining recommendation model...")
                  
    val als = new ALS()
       .setRank(12) // setting latent factors
      .setMaxIter(20) // setting maximum no. of iterations
      .setRegParam(0.01) //setting parameter for regularization
      .setImplicitPrefs(true) // setting implicit preferences for cold start user
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    
     // splitting the data set into test and training set
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))
      
    println("model trained fitting the data")
      
    // fitting the data in the model
    val model = als.fit(training)
    
    
    model.setColdStartStrategy("drop")
    
    
    // fitting the test data.
    val predictions = model.transform(test)

    
   // snippet to calculate the RMSE for the test set
   val evaluator = new RegressionEvaluator()
                  .setMetricName("rmse")
                  .setLabelCol("rating")
                  .setPredictionCol("prediction")

                  
// evaluating the error

val error:Double = evaluator.evaluate(predictions)
val rmse = math.sqrt(error)

println()
println(s"Using ALS the Root-mean-square error = $rmse")



 //Generate top 10 movie recommendations for each user
//val userRecs = model.recommendForAllUsers(10)

val userId:Int = args(0).toInt
val user = Seq(userId).toDF("userid")
val userSubsetRecs = model.recommendForUserSubset(user, 10)

println()
println(s"Top 10 movies for user id $userId are: -")

  for (userRecs <- userSubsetRecs) {
     val id = userRecs(0) 
      val myRecs = userRecs(1) // First column is userID, second is the recs
      val temp = myRecs.asInstanceOf[WrappedArray[Row]] // Tell Scala what it is
      for (rec <- temp) {
        val movie = rec.getAs[Int](0)
        val rating = rec.getAs[Float](1)
        val movieName = nameDict(movie)
        println(id,movieName,rating)
        
        
      }
   }

println()
println("Generate top 10 user recommendations")
val movieId:Int = args(0).toInt
val movie = Seq(movieId).toDF("userid")
val movieSubsetRecs = model.recommendForUserSubset(movie, 10)

println(s"Top 10 users for movie id $movieId are: -")

  for (movieRecs <- movieSubsetRecs) {
      val id = movieRecs(0).toString().toInt 
      val myRecs = movieRecs(1) // First column is userID, second is the recs
      val temp = myRecs.asInstanceOf[WrappedArray[Row]] // Tell Scala what it is
      for (rec <- temp) {
        val user = rec.getAs[Int](0)
        val rating = rec.getAs[Float](1)
        val movieName = nameDict(id)
        println(movieName,user,rating)
        
        
      }
   }
    
    spark.stop()

  }
}