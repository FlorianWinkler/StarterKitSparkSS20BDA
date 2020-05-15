package edu.campus02.iwi.spark.lv.classification.rf.covtype;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import edu.campus02.iwi.spark.lv.WinConfig;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import scala.collection.JavaConversions;
import spark.exercise.utils.PreprocessUtils;

public class TreesAndForestClassifier {

	public static final String BASE_PATH_TO_INPUT_DATA =
			"data/input/lv/covtype/data.csv";
	
	public static void main(String[] args) {

		WinConfig.setupEnv();

		//1) local spark conf and java spark context creation
	
	
		//2) load CSV file as Dataset<Row>
		
		
		//3) split the Dataset<Row> into training and test sets 80:20 
		
		
		//4) run trainSimpleDecisionTree method after writing it
		
		
		//7) run tuneSimpleDecisionTree method after writing it
	
		
		//8) BONUSTASK
		//trainRandomForest(trainingData, testData);
			
	}

	//4)
	public static void trainSimpleDecisionTree(Dataset<Row> training, Dataset<Row> test) {
		
		//4) build a pipeline with the following 2 stages

		//a) vector assembler
	
		//b) decision tree
		
		//5) configure the ML pipeline and use
		//training dataset for model fitting
		
		//6) make predictions
		
		// evaluate model's metrics
		// Confusion matrix
		//precision for classes/labels
			
		//print string representation of resulting tree...
		
	}
	
	//7)
	public static void tuneSimpleDecisionTree(Dataset<Row> training, Dataset<Row> test) {

		//a) build the same pipeline as above in step 5) with 2 stages 
		
		//b) build a grid in order to test different parameter combinations

		//c) config and run train validation split to choose the best set of params
		
		//use best model to predict classes for testSet 

		// evaluate model's metrics
		
		// Confusion matrix

		// precision and classes/labels
						
		// get best pipeline model after fitting & tuning
		
		// get the decision tree classifier model from the pipeline
		
	}
	
	//8)
	public static void trainRandomForest(Dataset<Row> training, Dataset<Row> test) {

		//build the same pipeline as above in step 5) with 2 stages  

		//9) use the RandomForestClassifier instead of a single DecisionTreeClassifier		

	}
	
	private static void printDecisionTreeModelSettings(DecisionTreeClassificationModel model) {
		System.out.println("model settings");
		System.out.println("  - depth: "+model.depth());
		System.out.println("  - nodes: "+model.numNodes());
		System.out.println("  - impurity: "+model.getImpurity());
		System.out.println();
		System.out.println("learned classification tree model:\n" + model.toDebugString());
	}
	
}
