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

	public static final String BASE_PATH_TO_INPUT_DATA = "data/input/lv/covtype/data.csv";
	public static void main(String[] args) {

		WinConfig.setupEnv();

		//1) local spark conf and java spark context creation
		SparkConf cnf = new SparkConf().setMaster("local").setAppName(TreesAndForestClassifier.class.getName());
		SparkSession spark = SparkSession.builder().config(cnf).getOrCreate();

		//2) load CSV file as Dataset<Row>
		Dataset<Row> rawData = spark.read()
				.option("header", true)
				.option("delimiter",",")
				.schema(PreprocessUtils.buildCovTypeCsvSchema())
				.csv(BASE_PATH_TO_INPUT_DATA);

		//3) split the Dataset<Row> into training and test sets 80:20
		Dataset<Row>[] splits = rawData.randomSplit(new double[]{0.80,0.20},1234L);
		Dataset<Row> trainingSet = splits[0].cache();
		Dataset<Row> testSet = splits[1].cache();


		//4) run trainSimpleDecisionTree method after writing it
		trainSimpleDecisionTree(trainingSet,testSet);

		//7) run tuneSimpleDecisionTree method after writing it
		tuneSimpleDecisionTree(trainingSet,testSet);

		//8) BONUSTASK
		trainRandomForest(trainingSet, testSet);
	}

	//4)
	public static void trainSimpleDecisionTree(Dataset<Row> training, Dataset<Row> test) {

		//4) build a pipeline with the following 2 stages
		String[] cols = Stream.of(training.columns()).limit(54).toArray(String[]::new);

		//a) vector assembler
		VectorAssembler veca = PreprocessUtils.configVectorAssembler(cols, "", "features");

		//b) decision tree
		DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features");

		//5) configure the ML pipeline and use
		//training dataset for model fitting
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {veca,dt});
		PipelineModel pmodel = pipeline.fit(training);

		//6) make predictions
		Dataset<Row> predictions = pmodel.transform(test);

		// evaluate model's metrics
		MulticlassMetrics metrics = new MulticlassMetrics( predictions.select("prediction", "label") );

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("Confusion matrix: \n" + confusion);

		//precision for classes
		for(int l=1;l <= 7; l++) {
			System.out.println("label "+l+" = p: "+metrics.precision(l));
		}
		System.out.println("accurracy = "+metrics.accuracy());

		//Show decision tree
		DecisionTreeClassificationModel treeModel = (DecisionTreeClassificationModel) (pmodel.stages()[1]);
		printDecisionTreeModelSettings(treeModel);

	}

	//7)
	public static void tuneSimpleDecisionTree(Dataset<Row> training, Dataset<Row> test) {

		//a) build the same pipeline as above in step 5) with 2 stages
		String[] cols = Stream.of(training.columns()).limit(54).toArray(String[]::new);
		VectorAssembler veca = PreprocessUtils.configVectorAssembler(cols, "", "features");
		DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features");
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {veca,dt});

		//b) build a grid in order to test different parameter combinations
		List<String> impurities = Arrays.asList(new String[]{"entropy","gini"});
		ParamMap[] paramGrid = new ParamGridBuilder().addGrid(dt.impurity(),
				JavaConversions.collectionAsScalaIterable(impurities))
				.addGrid(dt.maxBins(),new int[]{32, 64})
				.addGrid(dt.maxDepth(), new int[]{5, 7})
				.build();
		//c) config and run train validation split to choose the best set of params
		TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
				.setEstimator(pipeline)
				.setEvaluator(new MulticlassClassificationEvaluator())
				.setEstimatorParamMaps(paramGrid)
				.setTrainRatio(0.80);

		//run train validation split to choose the best set of params
		TrainValidationSplitModel vmodel = trainValidationSplit.fit(training);

		//use best model to predict classes for testSet
		// get best pipeline model after fitting & tuning
		// get the decision tree classifier model from the pipeline

		//Make predictions
		Dataset<Row> predictions = vmodel.transform(test);

		//Evaluate model's metrics
		MulticlassMetrics metrics = new MulticlassMetrics( predictions.select("prediction", "label") );

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("Confusion matrix: \n" + confusion);

		//precision for classes
		for(int l=1;l <= 7; l++) {
			System.out.println("label "+l+" = p: "+metrics.precision(l));
		}
		System.out.println("accurracy = "+metrics.accuracy());


		//Show decision tree
		DecisionTreeClassificationModel treeModel = (DecisionTreeClassificationModel) vmodel.bestModel();
		printDecisionTreeModelSettings(treeModel);

	}

	//8)
	public static void trainRandomForest(Dataset<Row> training, Dataset<Row> test) {

		//build the same pipeline as above in step 5) with 2 stages
		String[] cols = Stream.of(training.columns()).limit(54).toArray(String[]::new);
		VectorAssembler veca = PreprocessUtils.configVectorAssembler(cols, "", "features");

		//9) use the RandomForestClassifier instead of a single DecisionTreeClassifier
		RandomForestClassifier rf = new RandomForestClassifier()
				.setNumTrees(15)
				.setFeatureSubsetStrategy("auto")
				.setImpurity("entropy")
				.setMaxDepth(20)
				.setMaxBins(200)
				.setSeed(new Random().nextLong())
				.setLabelCol("label")
				.setFeaturesCol("features");

		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] {veca,rf});
		PipelineModel pmodel = pipeline.fit(training);

		//Make predictions
		Dataset<Row> predictions = pmodel.transform(test);

		//Evaluate model's metrics
		MulticlassMetrics metrics = new MulticlassMetrics( predictions.select("prediction", "label") );

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("Confusion matrix: \n" + confusion);

		//precision for classes
		for(int l=1;l <= 7; l++) {
			System.out.println("label "+l+" = p: "+metrics.precision(l));
		}
		System.out.println("accurracy = "+metrics.accuracy());

		//Show decision tree
		DecisionTreeClassificationModel treeModel = (DecisionTreeClassificationModel) (pmodel.stages()[1]);
		printDecisionTreeModelSettings(treeModel);



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