package bigdata

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

import org.apache.spark.graphx._
import scala.reflect.ClassTag

import org.apache.spark.graphx._
import org.apache.spark.graphx.lib._





object UserbasedCF {
 
  def parseNames(line: String) : Option[(VertexId, String)] = {
    var fields = line.split(",")
    if (fields.length > 1) {
      val heroID:Long = fields(0).trim().toLong
        return Some( fields(0).trim().toLong, fields(1).trim())
    } 
  
    return None // flatmap will just discard None results, and extract data from Some results.
  }
  
  
  /** Transform an input line from marvel-graph.txt into a List of Edges */
  def makeEdges(line: String) : List[Edge[Int]] = {
    import scala.collection.mutable.ListBuffer
    var edges = new ListBuffer[Edge[Int]]()
    val fields = line.split(",")
    val origin = fields(0)
    for (x <- 1 to (fields.length - 1)) {
      edges += Edge(origin.trim().toLong, fields(x).trim().toLong, 0)
    }
    
    return edges.toList
  }
  
  /** Our main function where the action happens */
  def main(args: Array[String]) {
    
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    
     // Create a SparkContext using every core of the local machine
    val sc = new SparkContext("local[*]", "UserbasedCF")
    
   // Build up our vertices
    val names = sc.textFile("../ratings.csv")
    val verts = names.flatMap(parseNames)
    
    // Build up our edges
    val lines = sc.textFile("../ratings.csv")
    val edges = lines.flatMap(makeEdges)    
    
 
    val default = "Nobody"
    val graph = Graph(verts, edges, default).cache()
    
    //
    println("\nTop 10 most-connected users are:")
    // 
    graph.degrees.join(verts).sortBy(_._2._1, ascending=false).take(10).foreach(f => {
      println(" The most connected user is: " + f._1.toString() + " with total connections : " + f._2._1)} )
      
      
      
      // Now let's do Breadth-First Search using the Pregel API
    println("\nComputing degrees of separation from user 318..")
    
       val root: VertexId = 318
    
    // Initialize each node with a distance of infinity, unless it's our starting point
    val initialGraph = graph.mapVertices((id, _) => if (id == root) 0.0 else Double.PositiveInfinity)

    // Using the Pregel API
    val bfs = initialGraph.pregel(Double.PositiveInfinity, 10)( 
        (id, attr, msg) => math.min(attr, msg), 
        
        // Our "send message" function propagates out to all neighbors
        // with the distance incremented by one.
        triplet => { 
          if (triplet.srcAttr != Double.PositiveInfinity) { 
            Iterator((triplet.dstId, triplet.srcAttr+1)) 
          } else { 
            Iterator.empty 
          } 
        }, 
        
        (a,b) => math.min(a,b) ).cache()
    
    // Print out the first 100 results:
    val dist = bfs.vertices.join(verts).take(100).filter(x => !x._2._1.toString().equals("Infinity"))
    
    for(x <- dist){
      val distance  = x._2._1
      val user = x._2._2
      println(s"The distance of user $user from 318 is $distance")
    }
    bfs.vertices.join(verts).take(1000).foreach(println)
    
    
    }
    
  }