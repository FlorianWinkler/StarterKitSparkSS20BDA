package edu.campus02.iwi.spark.lv.classification.nb.spam;

import edu.campus02.iwi.spark.lv.WinConfig;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import spark.exercise.utils.PreprocessUtils;

public class SpamDetectorBasic {

	public static final String BASE_PATH_TO_INPUT_DATA =
			"data/input/lv/emails/CSDMC2010_SPAM/";

	public static void main(String[] args) {

		WinConfig.setupEnv();

		//1) local spark conf and java spark context creation
	
		
		//2) register our UDF to read contents of a single mail msg
	
		
		//3) load CSV file containing files and lables as DataFrame
		
		
		//4) transform the data frame with a little help from
		// the previously registered UDF
		
		
		//5) split data into training (75%) vs. test set (25%)
		
		
		//6) build a pipeline with the following 3 stages
		//a) tokenize -> b) hashing term frequency -> c) naive bayes
		
		
		//7) configure the ML pipeline and use
		// trainingSet for model fitting
		
		
		//8) apply trained model on test set and compute
		// the corresponding confusion matrix
		// evaluate model's metrics
		// Confusion matrix
		// precision
		// total accuracy
	
		
	}
}
