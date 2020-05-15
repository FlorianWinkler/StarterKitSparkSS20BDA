package spark.exercise.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.jsoup.Jsoup;

public class PreprocessUtils {

	private final static Set<String> STOP_WORDS = 
			new HashSet<>(Arrays.asList(StopWordsRemover.loadDefaultStopWords("english")));
	
	public static List<PipelineStage> configStringIndexers(String[] icols, String suffix) {

		if (icols == null || icols.length == 0) {
			throw new IllegalArgumentException("please specify some columns to index");
		}

		List<PipelineStage> result = new ArrayList<>();
		for (String c : icols) {
			result.add(new StringIndexer()
						.setInputCol(c)
						.setOutputCol(c + suffix));
		}
		return result;
	}

	public static List<PipelineStage> configOneHotEncoders(String[] icols, String suffix1, String suffix2) {

		if (icols == null || icols.length == 0) {
			throw new IllegalArgumentException("please specify some columns to one-hot encode");
		}

		List<PipelineStage> result = new ArrayList<>();

		for (String c : icols) {
			result.add(new OneHotEncoder()
							.setDropLast(false)
							.setInputCol(c + suffix1)
							.setOutputCol(c + suffix2));
		}

		return result;

	}

	public static VectorAssembler configVectorAssembler(String[] icols, String suffix, String ocol) {
		return new VectorAssembler()
					.setInputCols(Arrays.stream(icols)
									.map(s -> s + suffix)
									.toArray(String[]::new))
					.setOutputCol(ocol);
	}

	public static UDF1<String, String> getUdfMailMsgReader(String pathToMsgs) {
		return filename -> {
			String raw = MailUtils.getMailBodyText(pathToMsgs + filename);
			if (raw != null) {
				return Jsoup.parse(raw).text();
			}
			return "";
		};
	}
	
	public static UDF1<String, String> getUdfTextCleaner50Shades() {
		return orig ->
				 Arrays.stream(orig.toLowerCase()
						.replaceAll("[^a-zA-Z0-9\\s]", "").split("\\s+"))
						.filter(w -> w.length() > 1 && !STOP_WORDS.contains(w))
						.collect(Collectors.joining(" "));
	}
	
	public static UDF1<SparseVector, Double> decodeOneHot() {
		return vec ->  Double.valueOf(vec.indices()[0]);
	}
	
	public static StructType buildCovTypeCsvSchema() {
		//shows for covtype example how to build 
		//a schema for CSV reading with types
		StructField[] fields = new StructField[55];
		for(int f = 0;f < fields.length; f++) {
			fields[f] = new StructField("f"+f, DataTypes.DoubleType, true, Metadata.empty());
		}
		fields[fields.length-1] = new StructField("label", DataTypes.DoubleType, true, Metadata.empty());
		return new StructType(fields);
	}

}
