package edu.campus02.iwi.spark.lv.classification.nb.lenses;

import java.util.Arrays;

import edu.campus02.iwi.spark.lv.WinConfig;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class ContactLensClassifier {

	public static void main(String[] args) {

		WinConfig.setupEnv();

		// TODO: 1) set spark config and init spark session
		
		// TODO: 2) read raw data as csv with corresponding options
		
		// TODO: 3) combine the separate columns as feature vector
		// all f0..f8 => 9 columns in total
		
		// TODO: 4) transform the raw data
		
		// TODO: 5) train a NaiveBayes model on the data
		
		// TODO: 6) predict a single hard-coded feature vector
		
		// TODO: 7) for demo only -> make predictions for all training vectors
		
		// TODO: 8) compute accuracy for complete dataset
		
		// TODO: 9) close spark session

	}

}