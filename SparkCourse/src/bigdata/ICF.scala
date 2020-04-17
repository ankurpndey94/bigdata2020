package bigdata


import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import scala.math.sqrt
import scala.collection.mutable.ListBuffer



object ICF {
  

  // For mapping movie id with movie names
  
  def loadMovieNames() : Map[Int, String] = {
    
    // To tackle character encoding issues
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

    var movieNames:Map[Int, String] = Map()
    
     val lines = Source.fromFile("../ml-100k/u.item").getLines()
     for (line <- lines) {
       var fields = line.split('|')
       if (fields.length > 1) {
        movieNames += (fields(0).toInt -> fields(1))
       }
     }
    
     return movieNames
  }
  
  type MovieRating = (Int, Double)
  type UserRatingPair = (Int, (MovieRating, MovieRating))
  def makePairs(userRatings:UserRatingPair) = {
    val movieRating1 = userRatings._2._1
    val movieRating2 = userRatings._2._2
    
    val movie1 = movieRating1._1
    val rating1 = movieRating1._2
    val movie2 = movieRating2._1
    val rating2 = movieRating2._2
    
    ((movie1, movie2), (rating1, rating2))
  }
  
  def filterDuplicates(userRatings:UserRatingPair):Boolean = {
    val movieRating1 = userRatings._2._1
    val movieRating2 = userRatings._2._2
    
    val movie1 = movieRating1._1
    val movie2 = movieRating2._1
    
    return movie1 < movie2
  }
  
  type RatingPair = (Double, Double)
  type RatingPairs = Iterable[RatingPair]
  
  def computeCosineSimilarity(ratingPairs:RatingPairs): (Double, Int) = {
    var numPairs:Int = 0
    var sum_x2:Double = 0.0
    var sum_y2:Double = 0.0
    var sum_xy:Double = 0.0
    
    for (pair <- ratingPairs) {
      val X = pair._1
      val Y = pair._2
      
      sum_x2 += X * X
      sum_y2 += Y * Y
      sum_xy += X * Y
      numPairs += 1
    }
    
    val num:Double = sum_xy
    val denom = sqrt(sum_x2) * sqrt(sum_y2)
    
    var score:Double = 0.0
    if (denom != 0) {
      score = num / denom
    }
    
    return (score, numPairs)
  }
  
  def main(args: Array[String]) {

    // To calculate run time of the algorithm 
    val t1 = System.nanoTime
    
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    

    val sc = new SparkContext("local[*]", "ItemBasedCF")
    
    println("\nLoading movie names...")
    val nameDict = loadMovieNames()
    
    val data = sc.textFile("../ml-100k/u.data")
    
    
    // splitting the data into test and train set 
    data.randomSplit(Array(0.8, 0.2))
    
    
    
 
    
    // creating list to store RMSE for each iteration
    var rmse1 = new ListBuffer[Double]()
    
      
      
     val ratings = data.map(l => l.split("\t")).map(l => (l(0).toInt, (l(1).toInt, l(2).toDouble)))
     val mergedRatings = ratings.join(ratings)   
     val uniqueRatings = mergedRatings.filter(filterDuplicates)
     val groups = uniqueRatings.map(makePairs)
     val pairRatings = groups.groupByKey()
     val similarities = pairRatings.mapValues(computeCosineSimilarity).cache()
    
     
   
    if (args.length > 0) {
      
      
      // setting hyper parameters for the algorithm
      val scoreThreshold = 0.97
      val coOccurenceThreshold = 50.0
      
      
    println("\nEnter the Movie ID: ")
    val movieID:Int = scala.io.StdIn.readInt()
        
      
      val filteredResults = similarities.filter( x =>
        {
          val pair = x._1
          val sim = x._2
          (pair._1 == movieID || pair._2 == movieID) && sim._1 > scoreThreshold && sim._2 > coOccurenceThreshold
        }
      )
      val results = filteredResults.map( x => (x._2, x._1)).sortByKey(false).take(10)
      val movieRatings = data.map(l => l.split("\t")).map(l => (l(1).toInt, l(2).toDouble))
      
      
  var totalRatings = movieRatings.mapValues(x => (x, 1)).reduceByKey((x,y) => (x._1 + y._1, x._2 + y._2))
  var averageRatings = totalRatings.mapValues(x => x._1 / x._2)
  var rate = averageRatings.filter(f => f._1 == movieID).collect()

  // generating the list for Storing RMSE for each iteration
  var rmse = new ListBuffer[Double]()

          
      for (result <- results) {
        val sim = result._1
        val pair = result._2
        var similarMovieID = pair._1
        if (similarMovieID == movieID) {
          similarMovieID = pair._2
        }
        var totalRatings = movieRatings.mapValues(x => (x, 1)).reduceByKey((x,y) => (x._1 + y._1, x._2 + y._2))
        var averageRatings = totalRatings.mapValues(x => x._1 / x._2)
        var fin = averageRatings.filter(f => f._1 == similarMovieID).collect()
        var rm:Double = 0.0
        for(i <- fin){
           for(j<- rate){
             rm = (i._2 * sim._1)
            } 
         }
        rmse += rm
       }
       val rmseList = rmse.toList
      var pred_rating:Double = rmseList.sum/rmseList.length
      pred_rating = pred_rating.toDouble
      
        var name = nameDict(movieID) 
       
       println(s"For movie $name predicted rating is $pred_rating" )

     }
      
    //displaying the Runtime of the algorithm
    val duration = (System.nanoTime - t1) / 1e9d
    println(s"The runtime of this algorithm is: $duration")
    
  }     
 }
